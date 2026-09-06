"""Final MIPIC (Matryoshka Representation Learning via Self-Distilled
Intra-Relational Alignment and Progressive Information Chaining) Wrapper.

This module implements the complete MIPIC objective as defined in mipic.pdf Eq. 18:
    L_MIPIC = alpha * L_MRL + (1 - alpha) * (L_SIA + L_PIC)

For TinyBERT-6L:
    alpha = 0.4
    L_MIPIC = 0.4 * L_MRL + 0.6 * (L_SIA + L_PIC)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from losses.mrl_loss import MRL_NESTED_DIMS, MRL_TEMPERATURE
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
)


@dataclass(frozen=True)
class MIPICConfig:
    """Configuration for the complete MIPIC model and training objective.

    alpha trades off MRL and MIPIC self-distillation mechanisms (SIA + PIC):
        L_MIPIC = alpha * L_MRL + (1 - alpha) * (L_SIA + L_PIC)
    For TinyBERT-6L, the paper specifies alpha = 0.4.
    """

    alpha: float = 0.4
    temperature: float = MRL_TEMPERATURE
    pic_temperature: float = 0.05
    pic_detach_target: bool = True
    matryoshka_dims: Tuple[int, ...] = MRL_NESTED_DIMS
    align_layers: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    # Eq. (4)/(5) attention softmax temperature; see MRLSIAPICConfig.
    sia_temperature: float = 1.0
    pic_pipeline_pairs: Tuple[Tuple[int, ...], ...] = TINYBERT_PIC_PIPELINE_PAIRS
    d_att: int = 64
    base_k: int = 64
    learning_rate: float = 2e-5
    sia_lr_multiplier: float = 2.0
    pic_lr_multiplier: float = 2.0
    weight_decay: float = 0.01

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in range [0.0, 1.0], got {self.alpha}.")


class TinyBERT_MIPIC(nn.Module):
    """Authoritative MIPIC model combining TinyBERT-6L with MRL, SIA, and PIC.

    Integrates:
    - Shared TinyBERT-6L encoder
    - MRL InfoNCE loss on CLS token over nested dimensions
    - SIA attention KL + CKA on hidden states across layers 1-6
    - PIC vertical progressive InfoNCE chaining across 5 checkpoints:
        (1, 16) -> (2, 32) -> (3, 64) -> (4, 256) -> (5, 512) -> (6, 768)

    Total objective:
        L_MIPIC = alpha * L_MRL + (1 - alpha) * (L_SIA + L_PIC)
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        config: Optional[MIPICConfig] = None,
        underlying_model: Optional[TinyBERT_MRL_SIA_PIC] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.config = config or MIPICConfig()

        if underlying_model is not None:
            self.core = underlying_model
        else:
            core_config = MRLSIAPICConfig(
                temperature=self.config.temperature,
                sia_temperature=self.config.sia_temperature,
                pic_temperature=self.config.pic_temperature,
                pic_detach_target=self.config.pic_detach_target,
                matryoshka_dims=self.config.matryoshka_dims,
                align_layers=self.config.align_layers,
                pic_pipeline_pairs=self.config.pic_pipeline_pairs,
                d_att=self.config.d_att,
                base_k=self.config.base_k,
                learning_rate=self.config.learning_rate,
                sia_lr_multiplier=self.config.sia_lr_multiplier,
                pic_lr_multiplier=self.config.pic_lr_multiplier,
                weight_decay=self.config.weight_decay,
            )
            self.core = TinyBERT_MRL_SIA_PIC(
                encoder=encoder,
                config=core_config,
                device=device,
            )

        if device is not None:
            self.core.to(device)

    @property
    def encoder(self) -> nn.Module:
        return self.core.encoder

    @property
    def mrl_loss_fn(self) -> nn.Module:
        return self.core.mrl_loss_fn

    @property
    def sia_loss_fn(self) -> nn.Module:
        return self.core.sia_loss_fn

    @property
    def pic_loss_fn(self) -> nn.Module:
        return self.core.pic_loss_fn

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        batch: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> object:
        """Forward pass for encoder inference or the DDP loss graph."""
        if batch is not None:
            return self.compute_losses(batch=batch)
        if input_ids is None:
            raise ValueError("input_ids is required when batch is not provided.")
        return self.core.forward(input_ids=input_ids, attention_mask=attention_mask)

    def compute_losses(
        self,
        first_input_ids: Optional[torch.Tensor] = None,
        first_attention_mask: Optional[torch.Tensor] = None,
        second_input_ids: Optional[torch.Tensor] = None,
        second_attention_mask: Optional[torch.Tensor] = None,
        *,
        batch: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute the complete MIPIC loss:
            L_MIPIC = alpha * L_MRL + (1 - alpha) * (L_SIA + L_PIC)
        """
        core_losses = self.core.compute_losses(
            first_input_ids=first_input_ids,
            first_attention_mask=first_attention_mask,
            second_input_ids=second_input_ids,
            second_attention_mask=second_attention_mask,
            batch=batch,
        )

        mrl_loss = core_losses["mrl_loss"]
        sia_loss = core_losses["sia_loss"]
        pic_loss = core_losses["pic_loss"]
        alpha = self.config.alpha

        # Paper Eq. 18: L_MIPIC = alpha * L_MRL + (1 - alpha) * (L_SIA + L_PIC)
        mipic_loss = alpha * mrl_loss + (1.0 - alpha) * (sia_loss + pic_loss)

        return {
            "loss": mipic_loss,
            "mipic_loss": mipic_loss,
            "mrl_loss": mrl_loss,
            "sia_loss": sia_loss,
            "pic_loss": pic_loss,
            "pic_pair_losses": core_losses["pic_pair_losses"],
            "sia_attn_loss": core_losses["sia_attn_loss"],
            "sia_cka_loss": core_losses["sia_cka_loss"],
            "mrl_logits": core_losses["mrl_logits"],
            "mrl_sia_loss": core_losses["mrl_sia_loss"],
            "first_output": core_losses.get("first_output"),
            "second_output": core_losses.get("second_output"),
        }


def create_mipic_optimizer(
    model: TinyBERT_MIPIC,
    lr: Optional[float] = None,
    sia_lr_multiplier: Optional[float] = None,
    pic_lr_multiplier: Optional[float] = None,
    weight_decay: Optional[float] = None,
) -> torch.optim.AdamW:
    """Create the MIPIC AdamW optimizer covering TinyBERT, SIA, and PIC parameters."""
    return create_mrl_sia_pic_optimizer(
        model=model.core,
        lr=lr if lr is not None else model.config.learning_rate,
        sia_lr_multiplier=sia_lr_multiplier if sia_lr_multiplier is not None else model.config.sia_lr_multiplier,
        pic_lr_multiplier=pic_lr_multiplier if pic_lr_multiplier is not None else model.config.pic_lr_multiplier,
        weight_decay=weight_decay if weight_decay is not None else model.config.weight_decay,
    )


def verify_mipic_optimizer(
    optimizer: torch.optim.Optimizer,
    model: TinyBERT_MIPIC,
) -> bool:
    """Programmatically verify that optimizer contains all TinyBERT, SIA, and PIC parameters."""
    return verify_mrl_sia_pic_optimizer(optimizer, model.core)

