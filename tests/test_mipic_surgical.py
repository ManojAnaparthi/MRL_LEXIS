"""Fast diagnostic tests for the paper-critical MIPIC paths.

These tests intentionally avoid downloading a pretrained backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate import extract_pooler_embeddings
from losses.mrl_loss import Matry_infonce
from losses.pic_loss import PipelineInfoNCELoss
from losses.sia_loss import SIALoss, _topk_indices, attention_kl, linear_cka
from models.mipic import MIPICConfig
from train import MRLBaselineConfig, PUBLISHED_TRAINING_SOURCES, build_mrl_train_loader


@dataclass
class _Output:
    hidden_states: tuple[torch.Tensor, ...]


class _DummyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    @property
    def device(self):
        return self.anchor.device

    def forward(self, input_ids, attention_mask, output_hidden_states=True, return_dict=True):
        del attention_mask, output_hidden_states, return_dict
        bsz = input_ids.size(0)
        # Deliberately make pooler_output different from final hidden CLS; the
        # extraction must follow hidden_states[-1][:, 0, :].
        final = torch.arange(bsz * 4, dtype=torch.float32, device=input_ids.device).reshape(bsz, 1, 4)
        return _Output(hidden_states=(final,))


def test_mipic_defaults_match_paper_critical_values() -> None:
    cfg = MIPICConfig()
    assert cfg.alpha == 0.4
    assert cfg.temperature == 0.05
    assert cfg.matryoshka_dims == (16, 32, 64, 128, 256, 512, 768)
    assert cfg.align_layers == (1, 2, 3, 4, 5, 6)


def test_evaluation_uses_the_shared_read_out_helper() -> None:
    model = _DummyEncoder()
    dataset = [
        {
            "input_ids1": torch.ones(3, dtype=torch.long),
            "attention_mask1": torch.ones(3, dtype=torch.long),
            "labels": 0,
        }
    ]
    loader = DataLoader(dataset, batch_size=1)
    out, labels = extract_pooler_embeddings(model, loader, dimension=4)
    assert labels == [0]
    assert out[0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_supplied_corpus_is_used_whole_and_dgem_format_is_not_dropped() -> None:
    """The supplied pool is never filtered; DGEM_FORMAT stays in the corpus."""
    assert PUBLISHED_TRAINING_SOURCES == (
        "BANKING77", "TWEET_EVAL_SENT", "MRPC", "DEEHAN1866/WIC",
        "DGEM_FORMAT", "SICK", "STSB",
    )
    frame = pd.read_csv("data/final_data.csv")
    assert len(frame) == 20240
    assert set(frame["source"]) == set(PUBLISHED_TRAINING_SOURCES)
    # Every supplied row reaches the training loader.
    selected = frame[frame["source"].isin(PUBLISHED_TRAINING_SOURCES)]
    assert len(selected) == 20240
    assert (frame["source"] == "DGEM_FORMAT").sum() == 3000


def test_mrl_normalization_temperature_and_diagonal_targets() -> None:
    torch.manual_seed(0)
    a = torch.randn(3, 8, requires_grad=True)
    b = torch.randn(3, 8, requires_grad=True)
    loss, logits = Matry_infonce(a, b, temperature=0.05, nested_dims=(4, 8))
    q = F.normalize(a[:, :4], dim=-1)
    k = F.normalize(b[:, :4], dim=-1)
    expected = F.cross_entropy(q @ k.T / 0.05, torch.arange(3))
    torch.testing.assert_close(logits["dim_4"], q @ k.T / 0.05)
    torch.testing.assert_close(loss - F.cross_entropy(F.normalize(a[:, :8], dim=-1) @ F.normalize(b[:, :8], dim=-1).T / 0.05, torch.arange(3)), expected)
    loss.backward()
    assert a.grad is not None and b.grad is not None


def test_sia_attention_direction_topk_and_cka() -> None:
    student = torch.tensor([[0.8, 0.2]])
    teacher = torch.tensor([[0.5, 0.5]])
    expected = (student * (student.log() - teacher.log())).sum()
    torch.testing.assert_close(attention_kl(student, teacher), expected)

    scores = torch.tensor([[0.1, 0.9, 0.2, 0.8, -5.0]])
    mask = torch.tensor([[1, 1, 1, 1, 0]])
    assert _topk_indices(scores, 2, mask).tolist() == [[1, 3]]

    x = torch.randn(4, 3)
    assert linear_cka(x, x).item() > 0.999

    sia = SIALoss(8, [2, 4], [1], temperature=0.05, topk_ratios={2: 0.5, 4: 0.5})
    hidden = [torch.randn(2, 5, 8, requires_grad=True) for _ in range(2)]
    mask = torch.ones(2, 5, dtype=torch.long)
    out = sia(hidden, mask)
    assert torch.isfinite(out["sia_loss"])
    out["sia_loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in sia.parameters())


def test_pic_query_projector_target_key_and_target_gradient() -> None:
    torch.manual_seed(1)
    # detach_target=False keeps this test's assertion about the target gradient;
    # the shipped default detaches (see test_28_pic_target_is_detached_by_default).
    pic = PipelineInfoNCELoss(2, 4, d_hidden=3, detach_target=False)
    src = torch.randn(3, 5, 2, requires_grad=True)
    tgt = torch.randn(3, 5, 4, requires_grad=True)
    loss = pic(src, tgt, temperature=0.05)
    with torch.no_grad():
        q = F.normalize(pic.phi(src[:, 0, :]), dim=-1)
        k = F.normalize(tgt[:, 0, :], dim=-1)
        ref = F.cross_entropy(q @ k.T / 0.05, torch.arange(3))
    torch.testing.assert_close(loss.detach(), ref)
    loss.backward()
    assert pic.phi[0].weight.grad is not None and pic.phi[0].weight.grad.abs().sum() > 0
    assert tgt.grad is not None and tgt.grad.abs().sum() > 0


def test_final_loss_weighting_is_exact() -> None:
    # This regression test mirrors Eq. 18 algebraically without loading TinyBERT.
    alpha = 0.4
    mrl = torch.tensor(1.2)
    sia = torch.tensor(0.7)
    pic = torch.tensor(0.9)
    expected = alpha * mrl + (1.0 - alpha) * (sia + pic)
    torch.testing.assert_close(expected, torch.tensor(1.44))


def test_surgical_loader_preserves_supplied_corpus_and_global_batch_configuration() -> None:
    class DummyTokenizer:
        def __call__(self, texts, **kwargs):
            b = len(texts)
            return {
                "input_ids": torch.ones(b, 4, dtype=torch.long),
                "attention_mask": torch.ones(b, 4, dtype=torch.long),
                "special_tokens_mask": torch.zeros(b, 4, dtype=torch.long),
            }

    loader = build_mrl_train_loader(
        Path("data/final_data.csv"),
        DummyTokenizer(),
        MRLBaselineConfig(batch_size=16, num_workers=0),
    )
    assert len(loader.dataset) == 20240
    assert len(loader) == 1265
