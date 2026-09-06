"""Combined TinyBERT-6L + MRL + PIC (w/o SIA) model and loss ablation.

This module implements the MRL + PIC ablation where SIA is removed:
    L_MRL_PIC = 0.4 * L_MRL + 0.6 * L_PIC

Uses:
- Shared TinyBERT-6L encoder
- MRL InfoNCE on final pooler outputs
- PIC vertical progressive InfoNCE on intermediate hidden states across 5 checkpoints:
    (1, 16) -> (2, 32) -> (3, 64) -> (4, 256) -> (5, 512) -> (6, 768)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from losses.mrl_loss import MRL_NESTED_DIMS, MRL_TEMPERATURE, MatryoshkaInfoNCELoss
from losses.pic_loss import PICModule
from models.mrl_sia_pic import (
    TINYBERT_PIC_CHECKPOINTS,
    TINYBERT_PIC_PIPELINE_PAIRS,
)
from models.tinybert import (
    TINYBERT_6L_HIDDEN_SIZE,
    TINYBERT_6L_NUM_HIDDEN_LAYERS,
    load_tinybert_6l,
    mrl_sentence_embedding,
)


@dataclass(frozen=True)
class MRLPICConfig:
    """Configuration for TinyBERT + MRL + PIC ablation training."""

    alpha: float = 0.4
    temperature: float = MRL_TEMPERATURE
    pic_temperature: float = 0.05
    # Stop-gradient on the deeper checkpoint z_{i+1}; see PipelineInfoNCELoss.
    pic_detach_target: bool = True
    matryoshka_dims: Tuple[int, ...] = MRL_NESTED_DIMS
    pic_pipeline_pairs: Tuple[Tuple[int, ...], ...] = TINYBERT_PIC_PIPELINE_PAIRS
    learning_rate: float = 2e-5
    pic_lr_multiplier: float = 1.0
    weight_decay: float = 0.01


class TinyBERT_MRL_PIC(nn.Module):
    """Composite model combining TinyBERT-6L with MRL and PIC losses (no SIA)."""

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        config: Optional[MRLPICConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.config = config or MRLPICConfig()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder, _ = load_tinybert_6l(device=device)

        if device is not None:
            self.encoder.to(device)


        # 1. Matryoshka InfoNCE Loss module
        self.mrl_loss_fn = MatryoshkaInfoNCELoss(
            temperature=self.config.temperature,
            nested_dims=self.config.matryoshka_dims,
        )

        # 2. Progressive Information Chaining (PIC) Loss module
        self.pic_loss_fn = PICModule(
            d_full=TINYBERT_6L_HIDDEN_SIZE,
            pipeline_pairs=[list(p) for p in self.config.pic_pipeline_pairs],
            temperature=self.config.pic_temperature,
            detach_target=self.config.pic_detach_target,
        )

        if device is not None:
            self.pic_loss_fn.to(device)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        batch: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> object:
        """Forward pass exposing TinyBERT encoder outputs."""
        if batch is not None:
            return self.compute_losses(batch=batch)
        if input_ids is None:
            raise ValueError("input_ids is required when batch is not provided.")
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    def compute_losses(
        self,
        first_input_ids: Optional[torch.Tensor] = None,
        first_attention_mask: Optional[torch.Tensor] = None,
        second_input_ids: Optional[torch.Tensor] = None,
        second_attention_mask: Optional[torch.Tensor] = None,
        *,
        batch: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute the combined MRL + PIC losses from two stochastic views.

        Objective:
            L_MRL_PIC = 0.4 * L_MRL + 0.6 * L_PIC
        """
        if batch is not None:
            first_input_ids = batch["input_ids1_stu"]
            first_attention_mask = batch["attention_mask1_stu"]
            second_input_ids = batch["input_ids2_stu"]
            second_attention_mask = batch["attention_mask2_stu"]

        if (
            first_input_ids is None
            or first_attention_mask is None
            or second_input_ids is None
            or second_attention_mask is None
        ):
            raise ValueError(
                "Either provide 'batch' or explicit 'first_input_ids', 'first_attention_mask', "
                "'second_input_ids', and 'second_attention_mask'."
            )

        # Forward pass 1 (shared TinyBERT encoder)
        first_output = self.encoder(
            input_ids=first_input_ids,
            attention_mask=first_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Forward pass 2 (shared TinyBERT encoder)
        second_output = self.encoder(
            input_ids=second_input_ids,
            attention_mask=second_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        if len(first_output.hidden_states) != TINYBERT_6L_NUM_HIDDEN_LAYERS + 1:
            raise ValueError(
                f"Expected {TINYBERT_6L_NUM_HIDDEN_LAYERS + 1} hidden states, "
                f"got {len(first_output.hidden_states)}."
            )

        # 1. Matryoshka InfoNCE Loss on the final hidden-state [CLS] vector.
        # Paper Sec. 2.2.1/3.2.1 define the representation as the encoder
        # hidden state (h_CLS), and the official notebook computes
        # Matry_infonce(S_last[:, 0, :], ...). Critically, evaluate.py probes
        # hidden_states[-1][:, 0, :], so training any arm on pooler_output would
        # optimise a different vector than the one that is measured.
        mrl_loss, mrl_logits = self.mrl_loss_fn(
            mrl_sentence_embedding(first_output),
            mrl_sentence_embedding(second_output),
        )

        # 2. PIC Loss on hidden states (5 progressive transitions)
        pic_out_1 = self.pic_loss_fn(list(first_output.hidden_states), first_attention_mask)
        pic_loss = pic_out_1["total_loss"]

        pic_pair_losses: Dict[str, torch.Tensor] = dict(pic_out_1["pair_losses"])

        # Retain a no-gradient second-view diagnostic only.
        with torch.no_grad():
            pic_out_2 = self.pic_loss_fn(list(second_output.hidden_states), second_attention_mask)

        # Weighted ablation loss: 0.4 * L_MRL + 0.6 * L_PIC
        total_loss = 0.4 * mrl_loss + 0.6 * pic_loss

        return {
            "loss": total_loss,
            "total_loss": total_loss,
            "mrl_loss": mrl_loss,
            "pic_loss": pic_loss,
            "pic_pair_losses": pic_pair_losses,
            "mrl_logits": mrl_logits,
            "first_output": first_output,
            "second_output": second_output,
        }


def create_mrl_pic_optimizer(
    model: TinyBERT_MRL_PIC,
    lr: Optional[float] = None,
    pic_lr_multiplier: Optional[float] = None,
    weight_decay: Optional[float] = None,
) -> torch.optim.AdamW:
    """Create the AdamW optimizer for TinyBERT + PIC (no SIA)."""
    learning_rate = lr if lr is not None else model.config.learning_rate
    pic_multiplier = pic_lr_multiplier if pic_lr_multiplier is not None else model.config.pic_lr_multiplier
    decay = weight_decay if weight_decay is not None else model.config.weight_decay

    param_groups = [
        {
            "params": list(model.encoder.parameters()),
            "lr": learning_rate,
            "weight_decay": decay,
        },
        {
            "params": list(model.pic_loss_fn.parameters()),
            "lr": learning_rate * pic_multiplier,
            "weight_decay": decay,
        },
    ]
    return torch.optim.AdamW(param_groups)


def verify_mrl_pic_optimizer(
    optimizer: torch.optim.Optimizer,
    model: TinyBERT_MRL_PIC,
) -> bool:
    """Verify that optimizer contains only TinyBERT and PIC parameters."""
    opt_params = {p for group in optimizer.param_groups for p in group["params"]}
    encoder_params = set(model.encoder.parameters())
    pic_params = set(model.pic_loss_fn.parameters())

    assert encoder_params.issubset(opt_params), "Encoder parameters missing from optimizer."
    assert pic_params.issubset(opt_params), "PIC parameters missing from optimizer."

    expected_params = encoder_params | pic_params
    assert opt_params == expected_params, (
        f"Optimizer parameter mismatch: {len(opt_params)} present vs "
        f"{len(expected_params)} expected."
    )
    assert len(pic_params) > 0, "No PIC parameters found."

    # Programmatically verify no SIA module is involved
    for name, _ in model.named_modules():
        assert "sia" not in name.lower(), f"Unexpected SIA module found in MRL+PIC: {name}"

    return True

