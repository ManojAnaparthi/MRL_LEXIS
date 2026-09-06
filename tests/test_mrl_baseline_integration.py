"""Integration verification for the uploaded MRL baseline only."""

import json
from pathlib import Path

import torch

from losses.mrl_loss import MRL_NESTED_DIMS, MatryoshkaInfoNCELoss
from models.tinybert import (
    TINYBERT_6L_HIDDEN_SIZE,
    TINYBERT_6L_NUM_HIDDEN_LAYERS,
    load_tinybert_6l,
)


def _notebook_matry_infonce():
    root = Path(__file__).resolve().parents[1]
    candidates = [root / "mipic.ipynb", root / "MIPIC_Kaggle.ipynb"]
    notebook_path = next((p for p in candidates if p.exists()), None)
    assert notebook_path is not None, "Missing official/reference MIPIC notebook"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "def Matry_infonce" in "".join(cell.get("source", []))
    )
    namespace = {"torch": torch, "F": torch.nn.functional}
    exec(source, namespace)
    return namespace["Matry_infonce"]


def test_refactored_mrl_matches_uploaded_baseline_and_trains_one_step() -> None:
    torch.manual_seed(42)
    model, tokenizer = load_tinybert_6l()
    model.train()

    first = tokenizer(
        ["A short baseline sentence.", "TinyBERT exposes hidden states."],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    second = tokenizer(
        ["A short baseline sentence.", "TinyBERT exposes hidden states."],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    first_output = model(**first, output_hidden_states=True, return_dict=True)
    second_output = model(**second, output_hidden_states=True, return_dict=True)

    assert model.config.hidden_size == TINYBERT_6L_HIDDEN_SIZE
    assert model.config.num_hidden_layers == TINYBERT_6L_NUM_HIDDEN_LAYERS
    assert len(model.encoder.layer) == TINYBERT_6L_NUM_HIDDEN_LAYERS
    assert len(first_output.hidden_states) == TINYBERT_6L_NUM_HIDDEN_LAYERS + 1
    assert all(state.shape[-1] == TINYBERT_6L_HIDDEN_SIZE for state in first_output.hidden_states)

    mrl_module = MatryoshkaInfoNCELoss()
    refactored_loss, _ = mrl_module(first_output.pooler_output, second_output.pooler_output)
    notebook_loss, _ = _notebook_matry_infonce()(
        first_output.pooler_output,
        second_output.pooler_output,
        temperature=0.05,
        nested_dims=list(MRL_NESTED_DIMS),
    )
    torch.testing.assert_close(refactored_loss, notebook_loss, rtol=1e-6, atol=1e-6)
    assert torch.isfinite(refactored_loss)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    optimizer.zero_grad(set_to_none=True)
    refactored_loss.backward()
    parameter = next(parameter for parameter in model.parameters() if parameter.grad is not None)
    before_step = parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(before_step, parameter.detach())
