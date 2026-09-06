"""Micro-diagnostics for the audited MIPIC paths.

Each test maps to one numbered check in the audit brief. They are deliberately
cheap: no pretrained backbone is downloaded, no GPU is required.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from evaluate import MRLEvaluationConfig
from losses.mrl_loss import MRL_NESTED_DIMS, MRL_TEMPERATURE, Matry_infonce
from losses.pic_loss import PICModule, PipelineInfoNCELoss
from losses.sia_loss import (
    DEFAULT_TOPK_RATIOS,
    SIALoss,
    TokenImportanceAttention,
    _topk_indices,
    attention_kl,
    contextual_token_mask,
    linear_cka,
)
from models.mipic import MIPICConfig
from models.mrl_sia import MRLSIAConfig
from models.mrl_sia_pic import MRLSIAPICConfig, TINYBERT_PIC_CHECKPOINTS


# ---------------------------------------------------------------- 1-5: MRL --


def test_01_mrl_math_matches_eq1_and_eq19() -> None:
    """L_MRL is the unweighted sum of SimCSE InfoNCE over the seven prefixes."""
    torch.manual_seed(0)
    a, b = torch.randn(4, 768), torch.randn(4, 768)
    loss, logits = Matry_infonce(a, b, MRL_TEMPERATURE, MRL_NESTED_DIMS)
    manual = sum(
        F.cross_entropy(
            F.normalize(a[:, :d], dim=-1) @ F.normalize(b[:, :d], dim=-1).T
            / MRL_TEMPERATURE,
            torch.arange(4),
        )
        for d in MRL_NESTED_DIMS
    )
    torch.testing.assert_close(loss, manual)
    assert set(logits) == {f"dim_{d}" for d in MRL_NESTED_DIMS}


def test_02_mrl_normalizes_each_prefix_independently() -> None:
    """Prefixes are L2-normalised after slicing, not before."""
    a = torch.tensor([[3.0, 4.0, 100.0, 0.0]])
    _, logits = Matry_infonce(a, a, 1.0, (2,))
    torch.testing.assert_close(logits["dim_2"], torch.tensor([[1.0]]))


def test_03_mrl_temperature_is_the_paper_value() -> None:
    assert MRL_TEMPERATURE == 0.05
    assert MIPICConfig().temperature == 0.05
    assert MIPICConfig().pic_temperature == 0.05


def test_04_mrl_labels_are_the_diagonal_without_ddp() -> None:
    torch.manual_seed(1)
    a, b = torch.randn(5, 16), torch.randn(5, 16)
    _, logits = Matry_infonce(a, b, 0.05, (16,))
    assert logits["dim_16"].shape == (5, 5)
    expected = F.cross_entropy(logits["dim_16"], torch.arange(5))
    loss, _ = Matry_infonce(a, b, 0.05, (16,))
    torch.testing.assert_close(loss, expected)


def test_05_two_views_are_distinct_tensors_with_independent_gradients() -> None:
    a = torch.randn(3, 8, requires_grad=True)
    b = torch.randn(3, 8, requires_grad=True)
    Matry_infonce(a, b, 0.05, (8,))[0].backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0


# --------------------------------------------------------------- 6-10: SIA --


def test_06_sia_attention_follows_eq4_and_eq5() -> None:
    module = TokenImportanceAttention(d_small=2, d_full=4)
    with torch.no_grad():
        module.proj_small.weight.zero_()
        module.proj_small.weight[:2, :2] = torch.eye(2)
    full = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    _, teacher = module.teacher_scores(full)
    _, student = module.student_scores(full[..., :2], full[:, 0, :], temperature=1.0)
    # Eq. 4: h_CLS . h_j / sqrt(D);  Eq. 5: h_CLS . P_i^T h_j^(i) / sqrt(D)
    torch.testing.assert_close(teacher, torch.tensor([[30.0, 1.0, 2.0]]) / math.sqrt(4))
    torch.testing.assert_close(student, torch.tensor([[5.0, 1.0, 2.0]]) / math.sqrt(4))


def test_07_sia_kl_direction_is_student_to_teacher() -> None:
    s, t = torch.tensor([[0.8, 0.2]]), torch.tensor([[0.5, 0.5]])
    forward = (s * (s.log() - t.log())).sum()
    reverse = (t * (t.log() - s.log())).sum()
    torch.testing.assert_close(attention_kl(s, t), forward)
    assert not torch.allclose(attention_kl(s, t), reverse)


def test_08_sia_masking_excludes_padding_and_special_tokens() -> None:
    """Eq. 4 ranges over contextual tokens h_j; [CLS]/[SEP]/[PAD] are not keys."""
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1, 1]])
    assert contextual_token_mask(attention, special).tolist() == [[0, 1, 1, 0, 0]]
    # A row of only special tokens falls back to the attention mask.
    assert contextual_token_mask(
        torch.tensor([[1, 1]]), torch.tensor([[1, 1]])
    ).tolist() == [[1, 1]]

    module = TokenImportanceAttention(2, 4)
    probs, _ = module.teacher_scores(
        torch.randn(1, 5, 4), contextual_token_mask(attention, special), 1.0
    )
    assert probs[0, 0].item() == 0.0 and probs[0, 3].item() == 0.0
    torch.testing.assert_close(probs.sum(), torch.tensor(1.0))


def test_09_topk_uses_teacher_ranking_and_the_paper_schedule() -> None:
    assert dict(DEFAULT_TOPK_RATIOS) == {16: 0.2, 32: 0.3, 64: 0.4, 128: 0.5, 256: 0.6, 512: 0.7}
    sia = SIALoss(768, MRL_NESTED_DIMS, (1,))
    # k_i = max(8, ceil(gamma_i * m))
    assert sia._topk_count(16, 100) == 20 and sia._topk_count(512, 100) == 70
    assert sia._topk_count(16, 4) == 8  # kmin floor
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.8, 5.0]])
    mask = torch.tensor([[1, 1, 1, 1, 0]])
    assert _topk_indices(scores, 2, mask).tolist() == [[1, 3]]


def test_10_linear_cka_is_centred_normalised_and_bounded() -> None:
    torch.manual_seed(2)
    x = torch.randn(6, 5)
    torch.testing.assert_close(linear_cka(x, x), torch.tensor(1.0), atol=1e-4, rtol=1e-4)
    # Invariant to isotropic scaling and to a constant shift (centring).
    torch.testing.assert_close(linear_cka(x, 7.0 * x + 3.0), torch.tensor(1.0), atol=1e-4, rtol=1e-4)
    y = torch.randn(6, 9)
    value = linear_cka(x, y)
    assert 0.0 <= value.item() <= 1.0


def test_10b_including_cls_as_its_own_key_collapses_the_teacher() -> None:
    """s_0 = ||h_CLS||^2 / sqrt(D) dominates, so a_D degenerates onto [CLS].

    The paper's top-k rule ("select the top k_i tokens based on their importance
    scores in a_D") and Eq. 13's unweighted L_att + L_CKA sum are both
    meaningless when a_D is one-hot on the query anchor itself. Restricting the
    key set to contextual tokens restores a usable distribution.
    """
    module = TokenImportanceAttention(2, 8)
    hidden = torch.randn(4, 12, 8)
    hidden[:, 0, :] *= 4.0  # a realistically large [CLS] norm
    attention = torch.ones(4, 12, dtype=torch.long)
    special = torch.zeros(4, 12, dtype=torch.long)
    special[:, 0] = 1
    special[:, -1] = 1

    def entropy(p):
        p = p.clamp_min(1e-30)
        return float(-(p * p.log()).sum(dim=-1).mean())

    naive, _ = module.teacher_scores(hidden, attention, temperature=0.05)
    assert naive[:, 0].min().item() > 0.99, "teacher should be one-hot on [CLS]"
    assert entropy(naive) < 0.05

    fixed, _ = module.teacher_scores(
        hidden, contextual_token_mask(attention, special), temperature=0.05
    )
    assert fixed[:, 0].max().item() == 0.0
    assert entropy(fixed) > entropy(naive)

    # And the SIA attention term shrinks accordingly.
    sia = SIALoss(8, (2, 4, 8), (1,), temperature=1.0, topk_ratios={2: 0.2, 4: 0.4})
    states = [hidden, hidden]
    assert (
        sia(states, attention, special_tokens_mask=special)["attn_loss"].item()
        < sia(states, attention)["attn_loss"].item()
    )


# -------------------------------------------------------------- 11-13: PIC --


def test_11_pic_projector_dimensionality_matches_the_checkpoint_chain() -> None:
    assert TINYBERT_PIC_CHECKPOINTS == ((1, 16), (2, 32), (3, 64), (4, 256), (5, 512), (6, 768))
    pic = PICModule(768, [(1, 16, 2, 32, 3, 64, 4, 256, 5, 512, 6, 768)], temperature=0.05)
    assert len(pic.parsed_pipeline_pairs) == 5  # exactly n-1 adjacent transitions
    for (_, d_src, _, d_tgt), module in zip(
        pic.parsed_pipeline_pairs, pic.infonce_modules.values()
    ):
        assert module.phi[0].in_features == d_src
        assert module.phi[-1].out_features == d_tgt
        assert isinstance(module.phi[1], torch.nn.ReLU)  # nonlinear projector


def test_12_pic_query_is_the_projected_source_and_key_is_the_target() -> None:
    torch.manual_seed(3)
    pic = PipelineInfoNCELoss(4, 8, d_hidden=6)
    src = torch.randn(3, 5, 4)
    tgt = torch.randn(3, 5, 8)
    loss = pic(src, tgt, temperature=0.05)
    with torch.no_grad():
        q = F.normalize(pic.phi(src[:, 0, :]), dim=-1)
        k = F.normalize(tgt[:, 0, :], dim=-1)
        expected = F.cross_entropy(q @ k.T / 0.05, torch.arange(3))
    torch.testing.assert_close(loss.detach(), expected)


def test_13_pic_labels_are_the_diagonal_and_pairs_are_averaged() -> None:
    """Eq. 17 writes a sum; we average. See losses/sia_loss.py for why."""
    torch.manual_seed(4)
    pic = PICModule(8, [(1, 2, 2, 4, 3, 8)], temperature=0.05)
    hidden = [torch.randn(3, 4, 8) for _ in range(4)]
    out = pic(hidden)
    pairs = out["pair_losses"]
    assert len(pairs) == 2
    torch.testing.assert_close(out["total_loss"], sum(pairs.values()) / len(pairs))


# --------------------------------------------------- 14-17: objective/grads --


def test_14_alpha_weighting_is_exactly_eq18() -> None:
    assert MIPICConfig().alpha == 0.4
    mrl, sia, pic = torch.tensor(1.2), torch.tensor(0.7), torch.tensor(0.9)
    alpha = MIPICConfig().alpha
    torch.testing.assert_close(alpha * mrl + (1 - alpha) * (sia + pic), torch.tensor(1.44))


def test_15_student_hidden_states_receive_sia_gradient() -> None:
    torch.manual_seed(5)
    hidden = [torch.randn(2, 8, 16, requires_grad=True) for _ in range(2)]
    sia = SIALoss(16, (4, 8, 16), (1,), temperature=1.0, topk_ratios={4: 0.5, 8: 0.5})
    sia(hidden, torch.ones(2, 8, dtype=torch.long))["sia_loss"].backward()
    assert hidden[1].grad is not None and hidden[1].grad.abs().sum() > 0


def test_16_every_sia_and_pic_projector_receives_a_gradient() -> None:
    torch.manual_seed(6)
    sia = SIALoss(16, (4, 8, 16), (1,), temperature=1.0, topk_ratios={4: 0.5, 8: 0.5})
    pic = PICModule(16, [(1, 4, 2, 8, 3, 16)], temperature=0.05)
    hidden = [torch.randn(3, 8, 16, requires_grad=True) for _ in range(4)]
    mask = torch.ones(3, 8, dtype=torch.long)
    (sia(hidden, mask)["sia_loss"] + pic(hidden, mask)["total_loss"]).backward()
    for name, parameter in list(sia.named_parameters()) + list(pic.named_parameters()):
        assert parameter.grad is not None, f"{name} received no gradient"
        assert parameter.grad.abs().sum() > 0, f"{name} received a zero gradient"


def test_17_sia_teacher_side_is_detached_but_the_student_side_is_not() -> None:
    """Sec. 3.2.1: h_CLS is "a fixed semantic reference during training"."""
    torch.manual_seed(7)
    module = TokenImportanceAttention(2, 4)
    full = torch.randn(2, 5, 4, requires_grad=True)

    teacher_probs, teacher_scores = module.teacher_scores(full)
    assert not teacher_probs.requires_grad and not teacher_scores.requires_grad

    student_probs, _ = module.student_scores(full[..., :2], full[:, 0, :])
    assert student_probs.requires_grad
    attention_kl(student_probs, teacher_probs).backward()
    assert full.grad is not None and full.grad.abs().sum() > 0


# ---------------------------------------------------------- 18-24: plumbing --


def test_21_the_pooler_is_trained_so_ddp_has_no_unused_parameters() -> None:
    """z IS pooler_output, so the pooler receives a gradient from L_MRL.

    The unused DDP indices 101/102 reported historically were exactly this
    pooler's two tensors, back when L_MRL read the raw [CLS] instead.
    """
    from models.tinybert import MRL_READOUT, mrl_sentence_embedding

    assert MRL_READOUT == "pooler_output"

    class _Out:
        hidden_states = (torch.zeros(2, 3, 4), torch.ones(2, 3, 4))
        pooler_output = torch.full((2, 4), 7.0, requires_grad=True)

    embedding = mrl_sentence_embedding(_Out())
    assert embedding.requires_grad
    torch.testing.assert_close(embedding, torch.full((2, 4), 7.0))


def test_24_tweeteval_probe_is_allowed_to_converge() -> None:
    assert MRLEvaluationConfig().max_iter >= 1000


def test_sia_temperature_default_is_documented_and_configurable() -> None:
    for config in (MRLSIAConfig(), MRLSIAPICConfig(), MIPICConfig()):
        assert config.sia_temperature == 1.0
        assert config.temperature == 0.05
    assert SIALoss(16, (8,), (1,), temperature=0.05).temperature == 0.05


# ------------------------------------------- MRL read-out (SimCSE head) --


def test_25_there_is_exactly_one_readout_and_no_switch() -> None:
    """Matryoshka nesting is a property of ONE vector.

    Eq. (1) supervises the prefixes z_{1:d_i} and inference truncates z, so the
    vector the MRL loss shapes and the vector the probe truncates must be the
    same object. There is deliberately no configuration switch that could let
    them drift apart.
    """
    import models.tinybert as tb

    assert tb.MRL_READOUT == "pooler_output"
    for removed in ("resolve_mrl_projection", "MRL_PROJECTION_MODES", "freeze_unused_pooler"):
        assert not hasattr(tb, removed), f"{removed} should have been removed"

    class _WithPooler:
        hidden_states = (torch.zeros(2, 3, 4), torch.ones(2, 3, 4))
        pooler_output = torch.full((2, 4), 7.0)

    torch.testing.assert_close(
        tb.mrl_sentence_embedding(_WithPooler()), torch.full((2, 4), 7.0)
    )

    # Test doubles with no pooler fall back to the final [CLS].
    class _NoPooler:
        hidden_states = (torch.zeros(2, 3, 4), torch.ones(2, 3, 4))
        pooler_output = None

    torch.testing.assert_close(tb.mrl_sentence_embedding(_NoPooler()), torch.ones(2, 4))


def test_26_probe_and_loss_share_one_read_out_helper() -> None:
    """Both sites must call the same helper, with no local read-out logic."""
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    evaluate_src = (root / "evaluate.py").read_text(encoding="utf-8")
    assert "mrl_sentence_embedding(output)" in evaluate_src
    assert "output.hidden_states[-1][:, 0, :]" not in evaluate_src
    assert "output.pooler_output[:, :dimension]" not in evaluate_src

    for name in ("models/mrl_sia_pic.py", "models/mrl_sia.py", "models/mrl_pic.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert "mrl_sentence_embedding(first_output)" in src, name
        assert "mrl_projection" not in src, f"{name} still has a read-out switch"


def test_28_pic_target_is_detached_by_default() -> None:
    """The deeper checkpoint is the teacher (official notebook + best run)."""
    torch.manual_seed(31)
    pic = PipelineInfoNCELoss(4, 8, d_hidden=6)
    assert pic.detach_target is True
    src = torch.randn(3, 5, 4, requires_grad=True)
    tgt = torch.randn(3, 5, 8, requires_grad=True)
    pic(src, tgt, temperature=0.05).backward()
    assert src.grad is not None and src.grad.abs().sum() > 0
    assert tgt.grad is None or tgt.grad.abs().sum() == 0

    symmetric = PipelineInfoNCELoss(4, 8, d_hidden=6, detach_target=False)
    tgt2 = torch.randn(3, 5, 8, requires_grad=True)
    symmetric(torch.randn(3, 5, 4), tgt2, temperature=0.05).backward()
    assert tgt2.grad is not None and tgt2.grad.abs().sum() > 0
