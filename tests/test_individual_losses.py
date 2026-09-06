"""Smoke tests for individual loss modules; no loss composition occurs here."""

import torch

from losses.mrl_loss import Matry_infonce
from losses.pic_loss import PICModule
from losses.sia_loss import SIALoss


def test_mrl_reference_loss() -> None:
    first = torch.randn(3, 768, requires_grad=True)
    second = torch.randn(3, 768, requires_grad=True)
    loss, logits = Matry_infonce(first, second, nested_dims=(16, 128, 768))
    assert loss.ndim == 0
    assert set(logits) == {"dim_16", "dim_128", "dim_768"}
    loss.backward()


def test_sia_standalone_loss() -> None:
    hidden_states = [torch.randn(2, 8, 768, requires_grad=True) for _ in range(7)]
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    loss_module = SIALoss(
        d_full=768,
        matryoshka_dims=(16, 128),
        align_layers=(2, 4),
        d_att=8,
        base_k=8,
    )
    result = loss_module(hidden_states, attention_mask)
    assert result["sia_loss"].ndim == 0
    assert torch.isfinite(result["sia_loss"])
    result["sia_loss"].backward()


def test_pic_standalone_loss() -> None:
    hidden_states = [torch.randn(3, 8, 768, requires_grad=True) for _ in range(7)]
    loss_module = PICModule(
        d_full=768,
        pipeline_pairs=[(1, 16, 3, 128, 5, 768)],
    )
    result = loss_module(hidden_states)
    assert result["total_loss"].ndim == 0
    assert torch.isfinite(result["total_loss"])
    result["total_loss"].backward()

def test_linear_cka_uses_float32_for_mps_backend() -> None:
    """MPS does not support float64; the CKA path must avoid requesting it."""
    import inspect
    from losses.sia_loss import linear_cka

    source = inspect.getsource(linear_cka)
    assert '.float()' in source
    assert 'torch.bmm(X, X.transpose(1, 2))' in source
    assert 'torch.float64' not in source
