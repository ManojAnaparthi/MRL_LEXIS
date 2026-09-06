"""Regression tests for paper-faithful MIPIC SIA behavior."""

import torch
import torch.nn.functional as F

import losses.sia_loss as sia_loss

from losses.sia_loss import (
    DEFAULT_TOPK_RATIOS,
    SIALoss,
    TokenImportanceAttention,
    attention_kl,
    _topk_indices,
)


def test_token_importance_matches_mipic_equations() -> None:
    torch.manual_seed(0)
    module = TokenImportanceAttention(d_small=2, d_full=4)

    # Make P_i deterministic: [x1, x2] -> [x1, x2, 0, 0].
    with torch.no_grad():
        module.proj_small.weight.zero_()
        module.proj_small.weight[:2, :2] = torch.eye(2)

    full = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0],
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0]]]
    )
    small = full[..., :2]

    teacher_probs, teacher_scores = module.teacher_scores(full)
    # Explicit dot products with the CLS anchor and sqrt(D) scaling.
    expected_teacher_scores = torch.tensor(
        [[30.0 / 2.0, 1.0 / 2.0, 2.0 / 2.0]]
    )
    assert torch.allclose(teacher_scores, expected_teacher_scores)
    assert torch.allclose(teacher_probs, F.softmax(expected_teacher_scores, dim=-1))

    # Student uses the same full-dimensional CLS query and P_i^T h_j^i as key.
    student_probs, student_scores = module.student_scores(
        small, full[:, 0, :], temperature=1.0
    )
    expected_student_scores = torch.tensor([[5.0 / 2.0, 1.0 / 2.0, 2.0 / 2.0]])
    assert torch.allclose(student_scores, expected_student_scores)
    assert torch.allclose(student_probs, F.softmax(expected_student_scores, dim=-1))


def test_attention_kl_is_student_to_teacher() -> None:
    student = torch.tensor([[0.8, 0.2]])
    teacher = torch.tensor([[0.5, 0.5]])
    expected = (student * (student.log() - teacher.log())).sum(dim=-1).mean()
    assert torch.allclose(attention_kl(student, teacher), expected)
    reverse = (teacher * (teacher.log() - student.log())).sum(dim=-1).mean()
    assert not torch.allclose(attention_kl(student, teacher), reverse)


def test_sia_uses_teacher_topk_and_paper_schedule() -> None:
    # The paper uses gamma=[.2,.3,.4,.5,.6,.7] for the first six prefixes.
    assert dict(DEFAULT_TOPK_RATIOS) == {
        16: 0.2, 32: 0.3, 64: 0.4, 128: 0.5, 256: 0.6, 512: 0.7
    }

    sia = SIALoss(
        d_full=768,
        matryoshka_dims=(16, 32, 64, 128, 256, 512, 768),
        align_layers=(1,),
    )
    assert sia.matryoshka_dims == [512, 256, 128, 64, 32, 16]
    assert sia._topk_count(16, 100) == 20
    assert sia._topk_count(32, 100) == 30
    assert sia._topk_count(512, 100) == 70
    assert all("768" not in name for name in sia.attn_modules)
    assert all(not hasattr(module, "W_Q_small") for module in sia.attn_modules.values())

    scores = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3]])
    indices = _topk_indices(scores, 2)
    assert indices.tolist() == [[1, 3]]


def test_sia_student_projection_receives_gradient_but_teacher_query_is_fixed() -> None:
    torch.manual_seed(1)
    module = TokenImportanceAttention(2, 4)
    full = torch.randn(2, 5, 4, requires_grad=True)
    small = full[..., :2]

    teacher_probs, _ = module.teacher_scores(full)
    student_probs, _ = module.student_scores(small, full[:, 0, :])
    loss = attention_kl(student_probs, teacher_probs)
    loss.backward()

    assert module.proj_small.weight.grad is not None
    assert module.proj_small.weight.grad.abs().sum() > 0


def test_sia_cka_uses_teacher_scores(monkeypatch) -> None:
    captured = []

    def fake_submatrix_cka(h_small, h_full, scores, k, mask=None, eps=1e-8, dtype=None):
        captured.append(scores.detach().clone())
        return h_small.new_zeros(())

    monkeypatch.setattr(sia_loss, "submatrix_cka", fake_submatrix_cka)

    torch.manual_seed(3)
    full = torch.randn(2, 6, 8)
    mask = torch.ones(2, 6, dtype=torch.long)
    module = SIALoss(8, [2, 4], [1], topk_ratios={2: 0.5, 4: 0.5})
    module([torch.zeros_like(full), full], mask)

    expected = torch.bmm(
        full[:, 0:1, :].detach(), full.detach().transpose(1, 2)
    ).squeeze(1) / (8 ** 0.5)
    assert len(captured) == 2
    assert all(torch.allclose(score, expected) for score in captured)


def test_attention_kl_is_finite_for_fp16_probabilities() -> None:
    student = torch.tensor([[1.0 - 1e-4, 1e-4]], dtype=torch.float16)
    teacher = torch.tensor([[0.5, 0.5]], dtype=torch.float16)
    value = attention_kl(student, teacher)
    assert torch.isfinite(value)


def test_sia_is_finite_for_fp16_hidden_states_under_cuda_autocast():
    """Low-temperature SIA must not emit +/-inf merely because encoder states are fp16."""
    torch.manual_seed(17)
    sia = SIALoss(768, [16, 32, 64, 128, 256, 512], [1], temperature=0.05, topk_ratios={
        16: 0.2, 32: 0.3, 64: 0.4, 128: 0.5, 256: 0.6, 512: 0.7,
    })
    hidden = [torch.randn(2, 256, 768, dtype=torch.float16) * 100 for _ in range(2)]
    mask = torch.ones(2, 256, dtype=torch.long)
    if torch.cuda.is_available():
        hidden = [x.cuda() for x in hidden]
        mask = mask.cuda()
        sia = sia.cuda()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = sia(hidden, mask)
    else:
        out = sia([x.float() for x in hidden], mask)
    assert torch.isfinite(out["sia_loss"])
    assert torch.isfinite(out["attn_loss"])
    assert torch.isfinite(out["cka_loss"])
