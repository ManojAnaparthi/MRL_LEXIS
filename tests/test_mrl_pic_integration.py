"""Integration verification tests for TinyBERT-6L + MRL + PIC ablation (Step 6 Phase A).

These tests verify:
1. Model construction works.
2. TinyBERT has 6 transformer layers and hidden size 768.
3. MRL dimensions include 16, 32, 64, 128, 256, 512, 768.
4. PIC contains exactly 5 transitions:
   16->32, 32->64, 64->256, 256->512, 512->768.
5. 128D is excluded from PIC.
6. MRL loss is finite.
7. PIC loss is finite.
8. Exact weighted loss formula: total = 0.4 * MRL + 0.6 * PIC.
9. Backward works.
10. TinyBERT receives non-zero gradients.
11. PIC parameters receive non-zero gradients.
12. Optimizer verification succeeds.
13. Optimizer step succeeds.
14. train_mrl_pic performs one training epoch/step successfully.
"""

from __future__ import annotations

import pytest
import torch

from models.mrl_pic import (
    MRLPICConfig,
    TinyBERT_MRL_PIC,
    create_mrl_pic_optimizer,
    verify_mrl_pic_optimizer,
)
from models.mrl_sia_pic import (
    TINYBERT_PIC_CHECKPOINTS,
    TINYBERT_PIC_PIPELINE_PAIRS,
)
from models.tinybert import (
    TINYBERT_6L_HIDDEN_SIZE,
    TINYBERT_6L_NUM_HIDDEN_LAYERS,
    load_tinybert_6l,
)
from train import train_mrl_pic


@pytest.fixture(scope="module")
def setup_mrl_pic_fixture():
    """Load model and small synthetic batch for testing."""
    torch.manual_seed(42)
    encoder, tokenizer = load_tinybert_6l()
    config = MRLPICConfig(
        alpha=0.4,
        temperature=0.05,
        pic_temperature=0.05,
        matryoshka_dims=(16, 32, 64, 128, 256, 512, 768),
        pic_pipeline_pairs=TINYBERT_PIC_PIPELINE_PAIRS,
    )
    wrapper = TinyBERT_MRL_PIC(encoder=encoder, config=config)
    wrapper.train()

    texts = [
        "Ablation testing MRL and PIC without SIA.",
        "TinyBERT verifies progressive depth chaining.",
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
    return wrapper, first, second, tokenizer


def test_1_and_2_model_construction_and_tinybert_specs(setup_mrl_pic_fixture) -> None:
    """TEST 1 & 2: Model construction works; TinyBERT has 6 layers, hidden size 768."""
    wrapper, _, _, _ = setup_mrl_pic_fixture
    assert isinstance(wrapper, TinyBERT_MRL_PIC)
    assert hasattr(wrapper, "encoder")
    assert hasattr(wrapper, "mrl_loss_fn")
    assert hasattr(wrapper, "pic_loss_fn")

    assert wrapper.encoder.config.num_hidden_layers == 6
    assert wrapper.encoder.config.hidden_size == 768
    assert wrapper.encoder.config.num_hidden_layers == TINYBERT_6L_NUM_HIDDEN_LAYERS
    assert wrapper.encoder.config.hidden_size == TINYBERT_6L_HIDDEN_SIZE


def test_3_to_5_dimensions_and_pic_transitions(setup_mrl_pic_fixture) -> None:
    """TESTS 3, 4, 5: MRL dimensions, exactly 5 PIC transitions, 128D excluded from PIC."""
    wrapper, _, _, _ = setup_mrl_pic_fixture

    # TEST 3: MRL dimensions include 16, 32, 64, 128, 256, 512, 768
    expected_mrl_dims = (16, 32, 64, 128, 256, 512, 768)
    assert tuple(wrapper.config.matryoshka_dims) == expected_mrl_dims

    # TEST 4: PIC contains exactly 5 transitions: 16->32, 32->64, 64->256, 256->512, 512->768
    parsed_pairs = wrapper.pic_loss_fn.parsed_pipeline_pairs
    assert len(parsed_pairs) == 5, f"Expected 5 PIC pairs, got {len(parsed_pairs)}."
    expected_pairs = [
        (1, 16, 2, 32),
        (2, 32, 3, 64),
        (3, 64, 4, 256),
        (4, 256, 5, 512),
        (5, 512, 6, 768),
    ]
    assert parsed_pairs == expected_pairs, (
        f"Parsed pairs {parsed_pairs} != expected {expected_pairs}"
    )

    # TEST 5: 128D is strictly excluded from PIC
    for layer, dim in TINYBERT_PIC_CHECKPOINTS:
        assert dim != 128, "128D found in TINYBERT_PIC_CHECKPOINTS"
    for _, d_src, _, d_tgt in parsed_pairs:
        assert d_src != 128 and d_tgt != 128, "128D found in parsed pipeline pairs"


def test_6_to_8_losses_finite_and_exact_formula(setup_mrl_pic_fixture) -> None:
    """TESTS 6, 7, 8: MRL and PIC losses are finite; total = 0.4 * MRL + 0.6 * PIC."""
    wrapper, first, second, _ = setup_mrl_pic_fixture
    losses = wrapper.compute_losses(
        first_input_ids=first["input_ids"],
        first_attention_mask=first["attention_mask"],
        second_input_ids=second["input_ids"],
        second_attention_mask=second["attention_mask"],
    )

    # TEST 6: MRL loss is finite
    assert torch.isfinite(losses["mrl_loss"])
    assert losses["mrl_loss"].item() > 0.0

    # TEST 7: PIC loss is finite
    assert torch.isfinite(losses["pic_loss"])
    assert losses["pic_loss"].item() > 0.0
    assert torch.isfinite(losses["total_loss"])

    # TEST 8: Exact weighted loss formula: total = 0.4 * MRL + 0.6 * PIC
    expected = 0.4 * losses["mrl_loss"] + 0.6 * losses["pic_loss"]
    torch.testing.assert_close(
        losses["total_loss"],
        expected,
        rtol=1e-6,
        atol=1e-6,
        msg="Total loss does not match 0.4 * MRL + 0.6 * PIC",
    )


def test_9_to_11_backward_and_gradients(setup_mrl_pic_fixture) -> None:
    """TESTS 9, 10, 11: Backward works; TinyBERT and PIC receive non-zero gradients."""
    wrapper, first, second, _ = setup_mrl_pic_fixture
    wrapper.zero_grad(set_to_none=True)

    losses = wrapper.compute_losses(
        first_input_ids=first["input_ids"],
        first_attention_mask=first["attention_mask"],
        second_input_ids=second["input_ids"],
        second_attention_mask=second["attention_mask"],
    )

    # TEST 9: Backward works
    losses["total_loss"].backward()

    # TEST 10: TinyBERT receives non-zero gradients
    encoder_params = [p for p in wrapper.encoder.parameters() if p.grad is not None]
    assert len(encoder_params) > 0, "No TinyBERT parameters received gradients."
    assert any(p.grad.abs().sum().item() > 0.0 for p in encoder_params), (
        "TinyBERT parameters received only zero gradients."
    )

    # TEST 11: PIC parameters receive non-zero gradients
    pic_params = [p for p in wrapper.pic_loss_fn.parameters() if p.grad is not None]
    assert len(pic_params) > 0, "No PIC parameters received gradients."
    assert all(p.grad.abs().sum().item() > 0.0 for p in pic_params), (
        "Some PIC parameters received zero gradient."
    )


def test_12_and_13_optimizer_verification_and_step(setup_mrl_pic_fixture) -> None:
    """TESTS 12 & 13: Optimizer verification succeeds; optimizer step succeeds."""
    wrapper, first, second, _ = setup_mrl_pic_fixture
    optimizer = create_mrl_pic_optimizer(wrapper)

    # TEST 12: Optimizer verification succeeds
    assert verify_mrl_pic_optimizer(optimizer, wrapper)

    # TEST 13: Optimizer step succeeds
    optimizer.zero_grad(set_to_none=True)
    losses = wrapper.compute_losses(
        first_input_ids=first["input_ids"],
        first_attention_mask=first["attention_mask"],
        second_input_ids=second["input_ids"],
        second_attention_mask=second["attention_mask"],
    )
    losses["total_loss"].backward()

    enc_param = next(p for p in wrapper.encoder.parameters() if p.grad is not None)
    pic_param = next(p for p in wrapper.pic_loss_fn.parameters() if p.grad is not None)

    enc_before = enc_param.detach().clone()
    pic_before = pic_param.detach().clone()

    optimizer.step()

    assert not torch.equal(enc_before, enc_param.detach()), "TinyBERT weights did not update."
    assert not torch.equal(pic_before, pic_param.detach()), "PIC weights did not update."


def test_14_train_mrl_pic_step_succeeds(setup_mrl_pic_fixture) -> None:
    """TEST 14: train_mrl_pic performs one training epoch/step successfully."""
    wrapper, first, second, _ = setup_mrl_pic_fixture
    batch_dict = {
        "input_ids1_stu": first["input_ids"],
        "attention_mask1_stu": first["attention_mask"],
        "input_ids2_stu": second["input_ids"],
        "attention_mask2_stu": second["attention_mask"],
    }
    history = train_mrl_pic(
        model=wrapper,
        train_loader=[batch_dict],
        device=torch.device("cpu"),
        epochs=1,
    )
    assert len(history) == 1
    assert "total_loss" in history[0]
    assert "mrl_loss" in history[0]
    assert "pic_loss" in history[0]
    assert history[0]["total_loss"] > 0.0
    assert history[0]["mrl_loss"] > 0.0
    assert history[0]["pic_loss"] > 0.0

