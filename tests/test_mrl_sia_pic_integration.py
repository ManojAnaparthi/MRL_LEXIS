"""Integration verification tests for TinyBERT-6L + MRL + SIA + PIC (Step 4).

These tests verify:
1. Model construction succeeds.
2. Correct TinyBERT layer-to-checkpoint mapping.
3. Correct PIC dimensions: 16 -> 32 -> 64 -> 256 -> 512 -> 768.
4. Verify 128-D is NOT included in PIC.
5. Exactly five PIC pair losses are produced.
6. PIC loss is finite.
7. MRL loss is finite.
8. SIA loss is finite.
9. Combined forward pass is finite.
10. PIC projectors have trainable parameters.
11. PIC projectors receive gradients after backward.
12. TinyBERT receives gradients.
13. SIA parameters receive gradients.
14. One optimizer step succeeds for TinyBERT, SIA, and PIC parameter groups.
"""

from __future__ import annotations

import pytest
import torch

from models.mrl_sia import TINYBERT_LAYER_TO_HIDDEN_STATE_INDEX
from models.mrl_sia_pic import (
    MRLSIAPICConfig,
    TINYBERT_PIC_CHECKPOINTS,
    TINYBERT_PIC_PIPELINE_PAIRS,
    TinyBERT_MRL_SIA_PIC,
    create_mrl_sia_pic_optimizer,
    verify_mrl_sia_pic_optimizer,
)
from models.tinybert import (
    TINYBERT_6L_HIDDEN_SIZE,
    TINYBERT_6L_NUM_HIDDEN_LAYERS,
    load_tinybert_6l,
)


@pytest.fixture(scope="module")
def setup_mrl_sia_pic_fixture():
    """Load model and small synthetic batch for testing."""
    torch.manual_seed(42)
    encoder, tokenizer = load_tinybert_6l()
    config = MRLSIAPICConfig(
        temperature=0.05,
        pic_temperature=0.05,
        matryoshka_dims=(16, 32, 64, 128, 256, 512, 768),
        align_layers=(1, 2, 3, 4, 5, 6),
        pic_pipeline_pairs=TINYBERT_PIC_PIPELINE_PAIRS,
        base_k=16,
    )
    wrapper = TinyBERT_MRL_SIA_PIC(encoder=encoder, config=config)
    wrapper.train()

    texts = [
        "Testing progressive information chaining with TinyBERT.",
        "MRL, SIA, and PIC combined without alpha formula.",
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


def test_1_model_construction_succeeds(setup_mrl_sia_pic_fixture) -> None:
    """TEST 1: Model construction succeeds with TinyBERT, MRL, SIA, and PIC."""
    wrapper, _, _ = setup_mrl_sia_pic_fixture
    assert isinstance(wrapper, TinyBERT_MRL_SIA_PIC)
    assert hasattr(wrapper, "encoder")
    assert hasattr(wrapper, "mrl_loss_fn")
    assert hasattr(wrapper, "sia_loss_fn")
    assert hasattr(wrapper, "pic_loss_fn")


def test_2_and_3_and_4_checkpoints_mapping_and_no_128(setup_mrl_sia_pic_fixture) -> None:
    """TESTS 2, 3, 4: Layer-to-checkpoint mapping, dimensions, and explicit omission of 128-D."""
    wrapper, _, _ = setup_mrl_sia_pic_fixture
    expected_checkpoints = ((1, 16), (2, 32), (3, 64), (4, 256), (5, 512), (6, 768))
    assert TINYBERT_PIC_CHECKPOINTS == expected_checkpoints, (
        f"Checkpoints {TINYBERT_PIC_CHECKPOINTS} do not match paper specification."
    )

    # Verify parsed pairs in PIC module
    parsed_pairs = wrapper.pic_loss_fn.parsed_pipeline_pairs
    assert len(parsed_pairs) == 5, f"Expected 5 pipeline pairs, got {len(parsed_pairs)}."

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

    # Verify dimensions are exactly 16 -> 32 -> 64 -> 256 -> 512 -> 768
    dimensions_in_pic = [dim for _, dim in expected_checkpoints]
    assert dimensions_in_pic == [16, 32, 64, 256, 512, 768]

    # Explicit check: dimension 128 is NOT included in PIC
    assert 128 not in dimensions_in_pic, "Dimension 128 must NOT be in the PIC chain for TinyBERT."
    for _, d_src, _, d_tgt in parsed_pairs:
        assert d_src != 128 and d_tgt != 128, "Dimension 128 found in PIC pipeline pair."

    # Layer mapping: hidden_states[0] is embedding, layers 1 to 6 are transformer layers
    layers_in_pic = [layer for layer, _ in expected_checkpoints]
    assert 0 not in layers_in_pic, "hidden_states[0] (embedding) must NOT be used in PIC."
    assert set(layers_in_pic) == {1, 2, 3, 4, 5, 6}


def test_5_pic_projector_parameters_trainable(setup_mrl_sia_pic_fixture) -> None:
    """TEST 9: PIC projectors have trainable parameters."""
    wrapper, _, _ = setup_mrl_sia_pic_fixture
    pic_params = list(wrapper.pic_loss_fn.parameters())
    assert len(pic_params) > 0, "No PIC parameters found."
    assert all(p.requires_grad for p in pic_params), "All PIC parameters should require grad."


def test_6_to_9_losses_finite_and_five_pic_pairs(setup_mrl_sia_pic_fixture) -> None:
    """TESTS 4, 5, 6, 7, 8: Forward pass produces 5 PIC pairs and all losses are finite."""
    wrapper, first, second = setup_mrl_sia_pic_fixture
    losses = wrapper.compute_losses(
        first_input_ids=first["input_ids"],
        first_attention_mask=first["attention_mask"],
        second_input_ids=second["input_ids"],
        second_attention_mask=second["attention_mask"],
    )

    # TEST 4: Exactly 5 PIC pair losses produced
    assert "pic_pair_losses" in losses
    assert len(losses["pic_pair_losses"]) == 5, (
        f"Expected 5 PIC pair losses, got {len(losses['pic_pair_losses'])}."
    )
    expected_keys = {
        "pipe_1_16_to_2_32",
        "pipe_2_32_to_3_64",
        "pipe_3_64_to_4_256",
        "pipe_4_256_to_5_512",
        "pipe_5_512_to_6_768",
    }
    assert set(losses["pic_pair_losses"].keys()) == expected_keys

    # Check each pair loss is finite
    for key, val in losses["pic_pair_losses"].items():
        assert torch.isfinite(val), f"Pair loss {key} is not finite."
        assert val.item() > 0.0, f"Pair loss {key} should be positive."

    # TEST 5: PIC loss is finite
    assert torch.isfinite(losses["pic_loss"])
    assert losses["pic_loss"].item() > 0.0

    # TEST 6: MRL loss is finite
    assert torch.isfinite(losses["mrl_loss"])
    assert losses["mrl_loss"].item() > 0.0

    # TEST 7: SIA loss is finite
    assert torch.isfinite(losses["sia_loss"])
    assert losses["sia_loss"].item() > 0.0

    # TEST 8: Combined forward loss is finite
    assert torch.isfinite(losses["total_loss"])
    assert losses["total_loss"].item() > 0.0


def test_10_to_13_backward_and_gradients(setup_mrl_sia_pic_fixture) -> None:
    """TESTS 10, 11: Backward succeeds; TinyBERT, SIA, and PIC projectors receive gradients."""
    wrapper, first, second = setup_mrl_sia_pic_fixture
    wrapper.zero_grad(set_to_none=True)

    losses = wrapper.compute_losses(
        first_input_ids=first["input_ids"],
        first_attention_mask=first["attention_mask"],
        second_input_ids=second["input_ids"],
        second_attention_mask=second["attention_mask"],
    )

    losses["total_loss"].backward()

    # TEST 10: PIC projectors receive non-zero gradients
    pic_params = [p for p in wrapper.pic_loss_fn.parameters() if p.grad is not None]
    assert len(pic_params) > 0, "No PIC parameters received gradients."
    assert all(p.grad.abs().sum().item() > 0.0 for p in pic_params), (
        "Some PIC projector parameters received zero gradient."
    )

    # TEST 11: TinyBERT receives non-zero gradients
    encoder_params = [p for p in wrapper.encoder.parameters() if p.grad is not None]
    assert len(encoder_params) > 0, "No TinyBERT parameters received gradients."
    assert any(p.grad.abs().sum().item() > 0.0 for p in encoder_params), (
        "TinyBERT parameters received only zero gradients."
    )

    # SIA parameters receive non-zero gradients
    sia_params = [p for p in wrapper.sia_loss_fn.parameters() if p.grad is not None]
    assert len(sia_params) > 0, "No SIA parameters received gradients."
    assert all(p.grad.abs().sum().item() > 0.0 for p in sia_params), (
        "Some SIA parameters received zero gradient."
    )


def test_14_optimizer_step_succeeds(setup_mrl_sia_pic_fixture) -> None:
    """Verify optimizer creation, parameter group verification, and parameter update step."""
    wrapper, first, second = setup_mrl_sia_pic_fixture
    optimizer = create_mrl_sia_pic_optimizer(wrapper)
    assert verify_mrl_sia_pic_optimizer(optimizer, wrapper)

    optimizer.zero_grad(set_to_none=True)
    losses = wrapper.compute_losses(
        first_input_ids=first["input_ids"],
        first_attention_mask=first["attention_mask"],
        second_input_ids=second["input_ids"],
        second_attention_mask=second["attention_mask"],
    )
    losses["total_loss"].backward()

    # Track pre-step parameters from each of the 3 groups
    enc_param = next(p for p in wrapper.encoder.parameters() if p.grad is not None)
    sia_param = next(p for p in wrapper.sia_loss_fn.parameters() if p.grad is not None)
    pic_param = next(p for p in wrapper.pic_loss_fn.parameters() if p.grad is not None)

    enc_before = enc_param.detach().clone()
    sia_before = sia_param.detach().clone()
    pic_before = pic_param.detach().clone()

    optimizer.step()

    # Verify weights updated
    assert not torch.equal(enc_before, enc_param.detach()), "TinyBERT weights did not change."
    assert not torch.equal(sia_before, sia_param.detach()), "SIA weights did not change."
    assert not torch.equal(pic_before, pic_param.detach()), "PIC weights did not change."

