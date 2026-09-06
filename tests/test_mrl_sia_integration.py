"""Integration verification tests for TinyBERT-6L + MRL + SIA.

These tests verify:
TEST 1: TinyBERT returns 7 hidden states.
TEST 2: SIA receives transformer layers 1–6 correctly.
TEST 3: SIA input tensor shapes are correct.
TEST 4: MRL loss is finite.
TEST 5: SIA loss is finite.
TEST 6: Total loss is finite.
TEST 7: Backward succeeds.
TEST 8: TinyBERT parameters receive non-zero gradients.
TEST 9: Trainable SIA parameters receive gradients.
TEST 10: PIC is not involved.
TEST 11: One optimizer step succeeds.
TEST 12: Total loss matches 0.4 * L_MRL + 0.6 * L_SIA within numerical tolerance.
"""

from __future__ import annotations

import pytest
import torch

from models.mrl_sia import (
    MRLSIAConfig,
    TINYBERT_LAYER_TO_HIDDEN_STATE_INDEX,
    TinyBERT_MRL_SIA,
    create_mrl_sia_optimizer,
    get_transformer_layer_outputs,
    verify_mrl_sia_optimizer,
)
from models.tinybert import (
    TINYBERT_6L_HIDDEN_SIZE,
    TINYBERT_6L_NUM_HIDDEN_LAYERS,
    load_tinybert_6l,
)


@pytest.fixture(scope="module")
def setup_model_and_inputs():
    """Load model, tokenizer, and small synthetic batch once for module tests."""
    torch.manual_seed(42)
    encoder, tokenizer = load_tinybert_6l()
    config = MRLSIAConfig(
        alpha=0.4,
        temperature=0.05,
        matryoshka_dims=(16, 32, 64, 128, 256, 512, 768),
        align_layers=(1, 2, 3, 4, 5, 6),
        d_att=64,
        base_k=16,
    )
    wrapper = TinyBERT_MRL_SIA(encoder=encoder, config=config)
    wrapper.train()

    texts = [
        "A short baseline sentence for testing.",
        "TinyBERT exposes intermediate hidden states for SIA alignment.",
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


def test_1_tinybert_returns_seven_hidden_states(setup_model_and_inputs) -> None:
    """TEST 1: TinyBERT returns 7 hidden states (index 0 is embedding, 1-6 are layers)."""
    wrapper, first, _ = setup_model_and_inputs
    output = wrapper.encoder(
        input_ids=first["input_ids"],
        attention_mask=first["attention_mask"],
        output_hidden_states=True,
        return_dict=True,
    )
    assert len(output.hidden_states) == 7, (
        f"Expected 7 hidden states (1 embedding + 6 transformer layers), "
        f"got {len(output.hidden_states)}."
    )
    for idx, state in enumerate(output.hidden_states):
        assert state.dim() == 3, f"hidden_states[{idx}] must be 3D"
        assert state.shape[-1] == TINYBERT_6L_HIDDEN_SIZE, (
            f"hidden_states[{idx}] hidden size should be {TINYBERT_6L_HIDDEN_SIZE}, "
            f"got {state.shape[-1]}"
        )


def test_2_sia_receives_transformer_layers_1_to_6(setup_model_and_inputs) -> None:
    """TEST 2: SIA receives transformer layers 1–6 correctly, excluding hidden_states[0]."""
    wrapper, first, _ = setup_model_and_inputs
    output = wrapper.encoder(
        input_ids=first["input_ids"],
        attention_mask=first["attention_mask"],
        output_hidden_states=True,
        return_dict=True,
    )

    # Verify align_layers explicitly covers layers 1 to 6
    assert list(wrapper.sia_loss_fn.align_layers) == [1, 2, 3, 4, 5, 6]
    assert 0 not in wrapper.sia_loss_fn.align_layers, (
        "hidden_states[0] (embedding output) must NOT be used as a transformer layer."
    )

    # Verify explicit layer mapping
    layer_outputs = get_transformer_layer_outputs(output.hidden_states)
    assert set(layer_outputs.keys()) == {1, 2, 3, 4, 5, 6}
    for layer_num, expected_idx in TINYBERT_LAYER_TO_HIDDEN_STATE_INDEX.items():
        assert torch.equal(layer_outputs[layer_num], output.hidden_states[expected_idx])


def test_3_sia_input_tensor_shapes(setup_model_and_inputs) -> None:
    """TEST 3: SIA input tensor shapes are valid [batch, seq_len, 768] and [batch, seq_len]."""
    wrapper, first, _ = setup_model_and_inputs
    output = wrapper.encoder(
        input_ids=first["input_ids"],
        attention_mask=first["attention_mask"],
        output_hidden_states=True,
        return_dict=True,
    )
    batch_size, seq_len = first["input_ids"].shape
    for layer in wrapper.sia_loss_fn.align_layers:
        state = output.hidden_states[layer]
        assert state.shape == (batch_size, seq_len, 768), (
            f"Layer {layer} hidden state shape {state.shape} != {(batch_size, seq_len, 768)}"
        )
    assert first["attention_mask"].shape == (batch_size, seq_len)


def test_4_to_6_losses_are_finite(setup_model_and_inputs) -> None:
    """TEST 4, 5, 6: MRL loss, SIA loss, and Total loss are finite and positive."""
    wrapper, first, second = setup_model_and_inputs
    losses = wrapper.compute_losses(
        first_input_ids=first["input_ids"],
        first_attention_mask=first["attention_mask"],
        second_input_ids=second["input_ids"],
        second_attention_mask=second["attention_mask"],
    )

    # TEST 4: MRL loss is finite
    assert torch.isfinite(losses["mrl_loss"]), "MRL loss is not finite."
    assert losses["mrl_loss"].item() > 0.0, "MRL loss should be strictly positive."

    # TEST 5: SIA loss is finite
    assert torch.isfinite(losses["sia_loss"]), "SIA loss is not finite."
    assert losses["sia_loss"].item() > 0.0, "SIA loss should be strictly positive."
    assert torch.isfinite(losses["sia_attn_loss"]), "SIA attention loss is not finite."
    assert torch.isfinite(losses["sia_cka_loss"]), "SIA CKA loss is not finite."

    # TEST 6: Total loss is finite
    assert torch.isfinite(losses["total_loss"]), "Total loss is not finite."
    assert losses["total_loss"].item() > 0.0, "Total loss should be strictly positive."


def test_7_to_9_backward_and_gradients(setup_model_and_inputs) -> None:
    """TEST 7, 8, 9: Backward succeeds, TinyBERT and SIA parameters receive gradients."""
    wrapper, first, second = setup_model_and_inputs
    wrapper.zero_grad(set_to_none=True)

    losses = wrapper.compute_losses(
        first_input_ids=first["input_ids"],
        first_attention_mask=first["attention_mask"],
        second_input_ids=second["input_ids"],
        second_attention_mask=second["attention_mask"],
    )

    # TEST 7: Backward succeeds
    total_loss = losses["total_loss"]
    total_loss.backward()

    # TEST 8: TinyBERT parameters receive non-zero gradients
    encoder_grads = [
        p.grad for p in wrapper.encoder.parameters() if p.grad is not None
    ]
    assert len(encoder_grads) > 0, "No TinyBERT parameters received gradients."
    assert any(g.abs().sum().item() > 0.0 for g in encoder_grads), (
        "TinyBERT parameters received only zero gradients."
    )

    # TEST 9: Trainable SIA parameters receive gradients
    sia_params = list(wrapper.sia_loss_fn.parameters())
    assert len(sia_params) > 0, "No trainable parameters found in SIA loss."
    assert all(p.grad is not None for p in sia_params), (
        "Some SIA trainable parameters have None gradient."
    )
    assert all(p.grad.abs().sum().item() > 0.0 for p in sia_params), (
        "Some SIA trainable parameters received zero gradient."
    )


def test_10_pic_is_not_involved(setup_model_and_inputs) -> None:
    """TEST 10: PIC is not involved in model, loss, or optimizer."""
    wrapper, _, _ = setup_model_and_inputs
    optimizer = create_mrl_sia_optimizer(wrapper)

    # Verify verify_mrl_sia_optimizer succeeds
    assert verify_mrl_sia_optimizer(optimizer, wrapper)

    # Verify no PIC modules exist in wrapper
    module_names = [name for name, _ in wrapper.named_modules()]
    for name in module_names:
        assert "pic" not in name.lower(), f"Unexpected PIC module found: {name}"

    # Verify no PIC parameters in optimizer
    all_opt_params = [p for g in optimizer.param_groups for p in g["params"]]
    all_wrapper_params = list(wrapper.parameters())
    assert len(all_opt_params) == len(all_wrapper_params)


def test_11_one_optimizer_step_succeeds(setup_model_and_inputs) -> None:
    """TEST 11: One optimizer step succeeds and updates both TinyBERT and SIA parameters."""
    wrapper, first, second = setup_model_and_inputs
    optimizer = create_mrl_sia_optimizer(wrapper)
    optimizer.zero_grad(set_to_none=True)

    losses = wrapper.compute_losses(
        first_input_ids=first["input_ids"],
        first_attention_mask=first["attention_mask"],
        second_input_ids=second["input_ids"],
        second_attention_mask=second["attention_mask"],
    )
    losses["total_loss"].backward()

    # Record weights before step
    encoder_param = next(p for p in wrapper.encoder.parameters() if p.grad is not None)
    sia_param = next(p for p in wrapper.sia_loss_fn.parameters() if p.grad is not None)
    enc_before = encoder_param.detach().clone()
    sia_before = sia_param.detach().clone()

    optimizer.step()

    # Assert weights changed
    assert not torch.equal(enc_before, encoder_param.detach()), (
        "TinyBERT parameter did not change after optimizer.step()."
    )
    assert not torch.equal(sia_before, sia_param.detach()), (
        "SIA parameter did not change after optimizer.step()."
    )


def test_12_total_loss_matches_weighted_combination(setup_model_and_inputs) -> None:
    """TEST 12: Total loss matches 0.4 * L_MRL + 0.6 * L_SIA within numerical tolerance."""
    wrapper, first, second = setup_model_and_inputs
    losses = wrapper.compute_losses(
        first_input_ids=first["input_ids"],
        first_attention_mask=first["attention_mask"],
        second_input_ids=second["input_ids"],
        second_attention_mask=second["attention_mask"],
    )

    expected = 0.4 * losses["mrl_loss"] + 0.6 * losses["sia_loss"]
    torch.testing.assert_close(
        losses["total_loss"],
        expected,
        rtol=1e-6,
        atol=1e-6,
        msg="Total loss does not match 0.4 * L_MRL + 0.6 * L_SIA",
    )

