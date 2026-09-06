"""Complete integration verification tests for TinyBERT-6L + MIPIC (Step 5).

Tests:
1. Final MIPIC wrapper constructs successfully.
2. alpha defaults to 0.4.
3. Invalid alpha values outside [0, 1] are rejected.
4. MRL loss is finite.
5. SIA loss is finite.
6. PIC loss is finite.
7. Final MIPIC loss is finite.
8. Exactly five PIC pair losses are still produced.
9. 128D is still excluded from PIC.
10. The final loss exactly matches: alpha * L_MRL + (1-alpha) * (L_SIA + L_PIC).
11. At alpha=0.4, verify the exact weighted formula (using torch.allclose).
12. At alpha=0, verify: L_MIPIC = L_SIA + L_PIC.
13. At alpha=1, verify: L_MIPIC = L_MRL.
14. Backward succeeds.
15. TinyBERT receives non-zero gradients.
16. SIA parameters receive non-zero gradients.
17. PIC projectors receive non-zero gradients.
18. Optimizer step succeeds.
"""

from __future__ import annotations

import pytest
import torch

from models.mipic import (
    MIPICConfig,
    TinyBERT_MIPIC,
    create_mipic_optimizer,
    verify_mipic_optimizer,
)
from models.mrl_sia_pic import (
    TINYBERT_PIC_CHECKPOINTS,
    TINYBERT_PIC_PIPELINE_PAIRS,
)
from models.tinybert import load_tinybert_6l


@pytest.fixture(scope="module")
def setup_mipic_fixture():
    """Load model and small synthetic batch for MIPIC testing."""
    torch.manual_seed(42)
    encoder, tokenizer = load_tinybert_6l()
    config = MIPICConfig(
        alpha=0.4,
        temperature=0.05,
        pic_temperature=0.05,
        matryoshka_dims=(16, 32, 64, 128, 256, 512, 768),
        align_layers=(1, 2, 3, 4, 5, 6),
        pic_pipeline_pairs=TINYBERT_PIC_PIPELINE_PAIRS,
        base_k=16,
    )
    wrapper = TinyBERT_MIPIC(encoder=encoder, config=config)
    wrapper.train()

    texts = [
        "MIPIC coordinates cross-dimensional and depth-wise alignment.",
        "TinyBERT evaluates MRL, SIA, and PIC in a unified framework.",
    ]
    first = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    second = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    return wrapper, first, second


def test_1_and_2_model_construction_and_alpha_default(setup_mipic_fixture) -> None:
    """TEST 1 & 2: Final MIPIC wrapper constructs successfully and alpha defaults to 0.4."""
    wrapper, _, _ = setup_mipic_fixture
    assert isinstance(wrapper, TinyBERT_MIPIC)
    assert hasattr(wrapper, "encoder")
    assert hasattr(wrapper, "mrl_loss_fn")
    assert hasattr(wrapper, "sia_loss_fn")
    assert hasattr(wrapper, "pic_loss_fn")

    default_cfg = MIPICConfig()
    assert default_cfg.alpha == 0.4, f"Default alpha should be 0.4, got {default_cfg.alpha}"
    assert wrapper.config.alpha == 0.4


def test_3_invalid_alpha_rejected() -> None:
    """TEST 3: Invalid alpha values outside [0, 1] are rejected with ValueError."""
    with pytest.raises(ValueError, match="alpha must be in range"):
        MIPICConfig(alpha=-0.1)

    with pytest.raises(ValueError, match="alpha must be in range"):
        MIPICConfig(alpha=1.5)

    with pytest.raises(ValueError, match="alpha must be in range"):
        MIPICConfig(alpha=-1e-5)

    with pytest.raises(ValueError, match="alpha must be in range"):
        MIPICConfig(alpha=1.0001)

    # Valid boundary values must not raise
    valid_zero = MIPICConfig(alpha=0.0)
    valid_one = MIPICConfig(alpha=1.0)
    assert valid_zero.alpha == 0.0
    assert valid_one.alpha == 1.0


def test_4_to_9_losses_finite_and_pic_structure(setup_mipic_fixture) -> None:
    """TESTS 4-9: MRL, SIA, PIC, and MIPIC losses are finite; 5 PIC pairs; no 128-D."""
    wrapper, first, second = setup_mipic_fixture
    losses = wrapper.compute_losses(
        first_input_ids=first["input_ids"],
        first_attention_mask=first["attention_mask"],
        second_input_ids=second["input_ids"],
        second_attention_mask=second["attention_mask"],
    )

    # TEST 4: MRL loss is finite
    assert torch.isfinite(losses["mrl_loss"])
    assert losses["mrl_loss"].item() > 0.0

    # TEST 5: SIA loss is finite
    assert torch.isfinite(losses["sia_loss"])
    assert losses["sia_loss"].item() > 0.0

    # TEST 6: PIC loss is finite
    assert torch.isfinite(losses["pic_loss"])
    assert losses["pic_loss"].item() > 0.0

    # TEST 7: Final MIPIC loss is finite
    assert torch.isfinite(losses["mipic_loss"])
    assert torch.isfinite(losses["loss"])
    assert losses["mipic_loss"].item() > 0.0
    assert losses["loss"].item() > 0.0
    assert torch.equal(losses["mipic_loss"], losses["loss"])

    # TEST 8: Exactly 5 PIC pair losses produced
    assert "pic_pair_losses" in losses
    assert len(losses["pic_pair_losses"]) == 5
    for key, val in losses["pic_pair_losses"].items():
        assert torch.isfinite(val)
        assert val.item() > 0.0

    # TEST 9: 128D is still excluded from PIC
    for _, dim in TINYBERT_PIC_CHECKPOINTS:
        assert dim != 128, "128D must NOT be in TINYBERT_PIC_CHECKPOINTS"
    for _, d_src, _, d_tgt in wrapper.pic_loss_fn.parsed_pipeline_pairs:
        assert d_src != 128 and d_tgt != 128, "128D found in PIC pipeline pairs."


def test_10_and_11_exact_loss_formula_at_alpha_0_4(setup_mipic_fixture) -> None:
    """TESTS 10 & 11: Final loss matches alpha * L_MRL + (1-alpha) * (L_SIA + L_PIC) at alpha=0.4."""
    wrapper, first, second = setup_mipic_fixture
    losses = wrapper.compute_losses(
        first_input_ids=first["input_ids"],
        first_attention_mask=first["attention_mask"],
        second_input_ids=second["input_ids"],
        second_attention_mask=second["attention_mask"],
    )

    mrl = losses["mrl_loss"]
    sia = losses["sia_loss"]
    pic = losses["pic_loss"]
    alpha = wrapper.config.alpha

    # TEST 10: General grouped formula
    expected_grouped = alpha * mrl + (1.0 - alpha) * (sia + pic)
    torch.testing.assert_close(
        losses["mipic_loss"],
        expected_grouped,
        rtol=1e-6,
        atol=1e-6,
        msg="mipic_loss does not match alpha * L_MRL + (1-alpha) * (L_SIA + L_PIC)",
    )

    # TEST 11: Exact 0.4 * L_MRL + 0.6 * (L_SIA + L_PIC)
    expected_alpha_0_4 = 0.4 * mrl + 0.6 * (sia + pic)
    torch.testing.assert_close(
        losses["mipic_loss"],
        expected_alpha_0_4,
        rtol=1e-6,
        atol=1e-6,
        msg="mipic_loss does not match 0.4 * L_MRL + 0.6 * (L_SIA + L_PIC)",
    )


def test_12_alpha_zero_boundary(setup_mipic_fixture) -> None:
    """TEST 12: At alpha=0, verify L_MIPIC = L_SIA + L_PIC."""
    wrapper, first, second = setup_mipic_fixture
    cfg_zero = MIPICConfig(
        alpha=0.0,
        temperature=wrapper.config.temperature,
        pic_temperature=wrapper.config.pic_temperature,
        matryoshka_dims=wrapper.config.matryoshka_dims,
        align_layers=wrapper.config.align_layers,
        pic_pipeline_pairs=wrapper.config.pic_pipeline_pairs,
        base_k=wrapper.config.base_k,
    )
    wrapper_zero = TinyBERT_MIPIC(underlying_model=wrapper.core, config=cfg_zero)

    losses = wrapper_zero.compute_losses(
        first_input_ids=first["input_ids"],
        first_attention_mask=first["attention_mask"],
        second_input_ids=second["input_ids"],
        second_attention_mask=second["attention_mask"],
    )

    expected = losses["sia_loss"] + losses["pic_loss"]
    torch.testing.assert_close(
        losses["mipic_loss"],
        expected,
        rtol=1e-6,
        atol=1e-6,
        msg="At alpha=0, L_MIPIC must equal L_SIA + L_PIC",
    )


def test_13_alpha_one_boundary(setup_mipic_fixture) -> None:
    """TEST 13: At alpha=1, verify L_MIPIC = L_MRL."""
    wrapper, first, second = setup_mipic_fixture
    cfg_one = MIPICConfig(
        alpha=1.0,
        temperature=wrapper.config.temperature,
        pic_temperature=wrapper.config.pic_temperature,
        matryoshka_dims=wrapper.config.matryoshka_dims,
        align_layers=wrapper.config.align_layers,
        pic_pipeline_pairs=wrapper.config.pic_pipeline_pairs,
        base_k=wrapper.config.base_k,
    )
    wrapper_one = TinyBERT_MIPIC(underlying_model=wrapper.core, config=cfg_one)

    losses = wrapper_one.compute_losses(
        first_input_ids=first["input_ids"],
        first_attention_mask=first["attention_mask"],
        second_input_ids=second["input_ids"],
        second_attention_mask=second["attention_mask"],
    )

    expected = losses["mrl_loss"]
    torch.testing.assert_close(
        losses["mipic_loss"],
        expected,
        rtol=1e-6,
        atol=1e-6,
        msg="At alpha=1, L_MIPIC must equal L_MRL",
    )


def test_14_to_17_backward_and_gradients(setup_mipic_fixture) -> None:
    """TESTS 14-17: Backward succeeds; TinyBERT, SIA, and PIC projectors receive gradients."""
    wrapper, first, second = setup_mipic_fixture
    wrapper.zero_grad(set_to_none=True)

    losses = wrapper.compute_losses(
        first_input_ids=first["input_ids"],
        first_attention_mask=first["attention_mask"],
        second_input_ids=second["input_ids"],
        second_attention_mask=second["attention_mask"],
    )

    # TEST 14: Backward succeeds
    losses["loss"].backward()

    # TEST 15: TinyBERT receives non-zero gradients
    encoder_params = [p for p in wrapper.encoder.parameters() if p.grad is not None]
    assert len(encoder_params) > 0, "No TinyBERT parameters received gradients."
    assert any(p.grad.abs().sum().item() > 0.0 for p in encoder_params), (
        "TinyBERT parameters received only zero gradients."
    )

    # TEST 16: SIA parameters receive non-zero gradients
    sia_params = [p for p in wrapper.sia_loss_fn.parameters() if p.grad is not None]
    assert len(sia_params) > 0, "No SIA parameters received gradients."
    assert all(p.grad.abs().sum().item() > 0.0 for p in sia_params), (
        "Some SIA parameters received zero gradient."
    )

    # TEST 17: PIC projectors receive non-zero gradients
    pic_params = [p for p in wrapper.pic_loss_fn.parameters() if p.grad is not None]
    assert len(pic_params) > 0, "No PIC parameters received gradients."
    assert all(p.grad.abs().sum().item() > 0.0 for p in pic_params), (
        "Some PIC projector parameters received zero gradient."
    )


def test_18_optimizer_verification_and_step(setup_mipic_fixture) -> None:
    """TEST 18: Optimizer step succeeds and modifies parameters across all 3 groups."""
    wrapper, first, second = setup_mipic_fixture
    optimizer = create_mipic_optimizer(wrapper)
    assert verify_mipic_optimizer(optimizer, wrapper)

    optimizer.zero_grad(set_to_none=True)
    losses = wrapper.compute_losses(
        first_input_ids=first["input_ids"],
        first_attention_mask=first["attention_mask"],
        second_input_ids=second["input_ids"],
        second_attention_mask=second["attention_mask"],
    )
    losses["loss"].backward()

    # Snapshot pre-step parameters
    enc_param = next(p for p in wrapper.encoder.parameters() if p.grad is not None)
    sia_param = next(p for p in wrapper.sia_loss_fn.parameters() if p.grad is not None)
    pic_param = next(p for p in wrapper.pic_loss_fn.parameters() if p.grad is not None)

    enc_before = enc_param.detach().clone()
    sia_before = sia_param.detach().clone()
    pic_before = pic_param.detach().clone()

    optimizer.step()

    # Assert weights changed
    assert not torch.equal(enc_before, enc_param.detach()), "TinyBERT parameters did not update."
    assert not torch.equal(sia_before, sia_param.detach()), "SIA parameters did not update."
    assert not torch.equal(pic_before, pic_param.detach()), "PIC parameters did not update."

