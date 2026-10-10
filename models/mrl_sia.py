"""Combined TinyBERT-6L + MRL + SIA model and loss integration.

This module combines the existing TinyBERT-6L encoder, the MRL InfoNCE loss,
and the standalone SIA loss into a unified, reusable training interface.
PIC is explicitly not included in this stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from losses.mrl_loss import MRL_NESTED_DIMS, MRL_TEMPERATURE, MatryoshkaInfoNCELoss
from losses.sia_loss import SIALoss
from models.tinybert import (
    TINYBERT_6L_HIDDEN_SIZE,
    TINYBERT_6L_NUM_HIDDEN_LAYERS,
    load_tinybert_6l,
    mrl_sentence_embedding,
)

# Explicit 1-based layer mapping from TinyBERT transformer layer index
# to the index in outputs.hidden_states (where index 0 is embedding output).
TINYBERT_LAYER_TO_HIDDEN_STATE_INDEX: Dict[int, int] = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
}


def get_transformer_layer_outputs(
    hidden_states: Sequence[torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """Map 1-based TinyBERT transformer layer numbers to their hidden state tensors."""
    if len(hidden_states) != TINYBERT_6L_NUM_HIDDEN_LAYERS + 1:
        raise ValueError(
            f"Expected {TINYBERT_6L_NUM_HIDDEN_LAYERS + 1} hidden states (embedding + 6 layers), "
            f"got {len(hidden_states)}."
        )
    return {
        layer: hidden_states[TINYBERT_LAYER_TO_HIDDEN_STATE_INDEX[layer]]
        for layer in sorted(TINYBERT_LAYER_TO_HIDDEN_STATE_INDEX.keys())
    }


@dataclass(frozen=True)
class MRLSIAConfig:
    """Configuration for combined TinyBERT + MRL + SIA training.

    Paper specification for TinyBERT-6L (MRL without PIC):
    - alpha = 0.4: total_loss = 0.4 * L_MRL + 0.6 * L_SIA
    - matryoshka_dims: [16, 32, 64, 128, 256, 512, 768]
    - align_layers: [1, 2, 3, 4, 5, 6] (all transformer layers)
    - temperature: 0.05
    """

    alpha: float = 0.4
    temperature: float = MRL_TEMPERATURE
    matryoshka_dims: Tuple[int, ...] = MRL_NESTED_DIMS
    align_layers: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    d_att: int = 64
    base_k: int = 64
    # Eq. (4)/(5) attention softmax temperature; see MRLSIAPICConfig.
    sia_temperature: float = 1.0
    learning_rate: float = 2e-5
    sia_lr_multiplier: float = 2.0
    weight_decay: float = 0.01


class TinyBERT_MRL_SIA(nn.Module):
    """Composite model combining TinyBERT-6L with MRL and SIA losses.

    One shared TinyBERT encoder processes two stochastic views.
    MRL operates on the final hidden-state [CLS] vector across nested dimensions.
    SIA operates over transformer layers 1 through 6, aligning token
    attention distributions and representations via top-k CKA.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        config: Optional[MRLSIAConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.config = config or MRLSIAConfig()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder, _ = load_tinybert_6l(device=device)

        if device is not None:
            self.encoder.to(device)


        # Matryoshka InfoNCE Loss module
        self.mrl_loss_fn = MatryoshkaInfoNCELoss(
            temperature=self.config.temperature,
            nested_dims=self.config.matryoshka_dims,
        )

        # Semantic Information Alignment (SIA) Loss module
        self.sia_loss_fn = SIALoss(
            d_full=TINYBERT_6L_HIDDEN_SIZE,
            matryoshka_dims=self.config.matryoshka_dims,
            align_layers=self.config.align_layers,
            d_att=self.config.d_att,
            base_k=self.config.base_k,
            temperature=self.config.sia_temperature,
            w_attn=1.0,
            w_cka=1.0,
            use_attention_kl=False,
        )

        if device is not None:
            self.sia_loss_fn.to(device)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        batch: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> object:
        """Standard encoder forward pass exposing pooler_output and hidden_states."""
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
        """Compute the combined MRL + SIA losses from two stochastic sentence views.

        Total loss:
            L_total = alpha * L_MRL + (1 - alpha) * L_SIA
        where alpha = 0.4 (default), yielding 0.4 * L_MRL + 0.6 * L_SIA.
        L_SIA is the average of SIA losses computed on both views.
        """
        first_special_tokens_mask = None
        second_special_tokens_mask = None
        if batch is not None:
            first_input_ids = batch["input_ids1_stu"]
            first_attention_mask = batch["attention_mask1_stu"]
            second_input_ids = batch["input_ids2_stu"]
            second_attention_mask = batch["attention_mask2_stu"]
            first_special_tokens_mask = batch.get("special_tokens_mask1_stu")
            second_special_tokens_mask = batch.get("special_tokens_mask2_stu")

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

        # Verify hidden_states length matches expectation: 1 embedding + 6 layers = 7
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

        # 2. SIA Loss on hidden states (layers 1-6) and attention masks for both views
        sia_out_1 = self.sia_loss_fn(
            first_output.hidden_states,
            first_attention_mask,
            special_tokens_mask=first_special_tokens_mask,
        )
        sia_loss = sia_out_1["sia_loss"]
        sia_attn_loss = sia_out_1["attn_loss"]
        sia_cka_loss = sia_out_1["cka_loss"]

        # The second stochastic view is used by MRL; retain a no-gradient
        # SIA diagnostic without contributing a second SIA training term.
        with torch.no_grad():
            sia_out_2 = self.sia_loss_fn(
                second_output.hidden_states,
                second_attention_mask,
                special_tokens_mask=second_special_tokens_mask,
            )

        # 3. Composite loss: 0.4 * L_MRL + 0.6 * L_SIA (alpha = 0.4)
        alpha = self.config.alpha
        total_loss = alpha * mrl_loss + (1.0 - alpha) * sia_loss

        return {
            "total_loss": total_loss,
            "mrl_loss": mrl_loss,
            "sia_loss": sia_loss,
            "sia_attn_loss": sia_attn_loss,
            "sia_cka_loss": sia_cka_loss,
            "mrl_logits": mrl_logits,
            "first_output": first_output,
            "second_output": second_output,
        }


def create_mrl_sia_optimizer(
    model: TinyBERT_MRL_SIA,
    lr: Optional[float] = None,
    sia_lr_multiplier: Optional[float] = None,
    weight_decay: Optional[float] = None,
) -> torch.optim.AdamW:
    """Create the AdamW optimizer with distinct learning rates for backbone and SIA."""
    learning_rate = lr if lr is not None else model.config.learning_rate
    multiplier = sia_lr_multiplier if sia_lr_multiplier is not None else model.config.sia_lr_multiplier
    decay = weight_decay if weight_decay is not None else model.config.weight_decay

    param_groups = [
        {
            "params": list(model.encoder.parameters()),
            "lr": learning_rate,
            "weight_decay": decay,
        },
        {
            "params": list(model.sia_loss_fn.parameters()),
            "lr": learning_rate * multiplier,
            "weight_decay": decay,
        },
    ]
    return torch.optim.AdamW(param_groups)


def verify_mrl_sia_optimizer(
    optimizer: torch.optim.Optimizer,
    model: TinyBERT_MRL_SIA,
) -> bool:
    """Programmatically verify optimizer parameter groups and exclude PIC."""
    opt_params = {p for group in optimizer.param_groups for p in group["params"]}
    encoder_params = set(model.encoder.parameters())
    sia_params = set(model.sia_loss_fn.parameters())

    assert encoder_params.issubset(opt_params), "Encoder parameters are missing from optimizer."
    assert sia_params.issubset(opt_params), "SIA parameters are missing from optimizer."

    expected_params = encoder_params | sia_params
    assert opt_params == expected_params, (
        f"Optimizer contains unexpected parameters: {len(opt_params)} present vs "
        f"{len(expected_params)} expected."
    )

    # Verify no PIC parameters or modules are involved
    for name, _ in model.named_modules():
        assert "pic" not in name.lower(), f"Unexpected PIC module found: {name}"

    return True

