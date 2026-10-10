"""Combined TinyBERT-6L + MRL + SIA + PIC model and loss integration.

This module integrates:
- TinyBERT-6L backbone encoder
- Matryoshka InfoNCE loss (MRL, horizontal across dimensions)
- Semantic Information Alignment loss (SIA, token-level across layers 1-6)
- Progressive Information Chaining loss (PIC, vertical depth-wise across 5
  checkpoint transitions)

PIC checkpoints follow mipic.pdf Section A.5 for TinyBERT-6L:
    C = {(16, 1), (32, 2), (64, 3), (256, 4), (512, 5), (768, 6)}
Dimension 128 is intentionally omitted from the PIC chain.

MRL is computed from the two stochastic views.
SIA and PIC are computed from the first stochastic view only, matching the
paper's objective decomposition. The second-view SIA/PIC outputs are retained
as no-gradient diagnostics for API compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from losses.mrl_loss import (
    MRL_NESTED_DIMS,
    MRL_TEMPERATURE,
    MatryoshkaInfoNCELoss,
)
from losses.pic_loss import PICModule
from losses.sia_loss import SIALoss
from models.mrl_sia import TINYBERT_LAYER_TO_HIDDEN_STATE_INDEX
from models.tinybert import (
    TINYBERT_6L_HIDDEN_SIZE,
    TINYBERT_6L_NUM_HIDDEN_LAYERS,
    load_tinybert_6l,
    mrl_sentence_embedding,
)


# Authoritative TinyBERT PIC checkpoints from mipic.pdf Section A.5:
# C = {(16, 1), (32, 2), (64, 3), (256, 4), (512, 5), (768, 6)}
# Represented as (layer, dim) tuples.
TINYBERT_PIC_CHECKPOINTS: Tuple[Tuple[int, int], ...] = (
    (1, 16),
    (2, 32),
    (3, 64),
    (4, 256),
    (5, 512),
    (6, 768),
)

# Pipeline configuration tuple:
# (layer_1, dim_1, layer_2, dim_2, ...)
TINYBERT_PIC_PIPELINE_PAIRS: Tuple[Tuple[int, ...], ...] = (
    (1, 16, 2, 32, 3, 64, 4, 256, 5, 512, 6, 768),
)


@dataclass(frozen=True)
class MRLSIAPICConfig:
    """Configuration for combined TinyBERT + MRL + SIA + PIC training."""

    # MRL configuration
    temperature: float = MRL_TEMPERATURE
    matryoshka_dims: Tuple[int, ...] = MRL_NESTED_DIMS

    # SIA configuration
    align_layers: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    d_att: int = 64
    base_k: int = 64
    # Softmax temperature of Eq. (4)/(5). The paper reuses the symbol tau, but
    # Table 7's "Temperature (tau) = 0.05" is introduced in A.3 as the
    # *L_SimCSE* temperature; Eqs. (4)/(5) are a scaled-dot-product attention
    # that is already divided by sqrt(D). The official implementation notebook
    # instantiates its alignment module with temperature=1.0, so that value is
    # used here. Set to 0.05 for the literal-symbol reading.
    sia_temperature: float = 1.0

    # PIC configuration (mipic.pdf Section A.5)
    pic_temperature: float = 0.05
    # Stop-gradient on the deeper checkpoint z_{i+1}; see PipelineInfoNCELoss.
    pic_detach_target: bool = True
    pic_pipeline_pairs: Tuple[Tuple[int, ...], ...] = TINYBERT_PIC_PIPELINE_PAIRS

    # Optimizer configuration
    learning_rate: float = 2e-5
    sia_lr_multiplier: float = 2.0
    pic_lr_multiplier: float = 2.0
    weight_decay: float = 0.01


class TinyBERT_MRL_SIA_PIC(nn.Module):
    """Composite model combining TinyBERT-6L with MRL, SIA, and PIC losses.

    One shared TinyBERT encoder processes two stochastic views.

    - MRL operates on the final hidden-state [CLS] embeddings across nested
      dimensions using both views.
    - SIA operates over transformer layers 1-6 using both stochastic views.
    - PIC operates over the progressive depth-wise checkpoint transitions
      using both stochastic views:
        (L1, D16) -> (L2, D32) -> (L3, D64)
        -> (L4, D256) -> (L5, D512) -> (L6, D768)

    SIA/PIC are computed for both stochastic views and averaged, following the
    structural pattern of the supplied implementation notebook.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        config: Optional[MRLSIAPICConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.config = config or MRLSIAPICConfig()

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

        # 2. Semantic Information Alignment (SIA) Loss module
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

        # 3. Progressive Information Chaining (PIC) Loss module
        self.pic_loss_fn = PICModule(
            d_full=TINYBERT_6L_HIDDEN_SIZE,
            pipeline_pairs=[list(p) for p in self.config.pic_pipeline_pairs],
            temperature=self.config.pic_temperature,
            detach_target=self.config.pic_detach_target,
        )

        if device is not None:
            self.sia_loss_fn.to(device)
            self.pic_loss_fn.to(device)

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
        """Compute MRL, SIA, and PIC over the two stochastic views.

        The execution structure follows the official implementation: both
        dropout views are forwarded, alignment losses are averaged across the
        two views, and MRL uses the two view-level CLS embeddings. The actual
        formulas/parameters remain paper-faithful.

        The final Eq. (18) weighting is applied by ``TinyBERT_MIPIC``.
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
                "Either provide 'batch' or explicit 'first_input_ids', "
                "'first_attention_mask', 'second_input_ids', and "
                "'second_attention_mask'."
            )

        # Official-implementation structure: two independent stochastic
        # dropout views from the same shared encoder.
        first_output = self.encoder(
            input_ids=first_input_ids,
            attention_mask=first_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        second_output = self.encoder(
            input_ids=second_input_ids,
            attention_mask=second_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        expected_hidden_states = TINYBERT_6L_NUM_HIDDEN_LAYERS + 1
        for name, output in (("first", first_output), ("second", second_output)):
            if len(output.hidden_states) != expected_hidden_states:
                raise ValueError(
                    f"Expected {expected_hidden_states} hidden states for {name} view, "
                    f"got {len(output.hidden_states)}."
                )

        # 1. MRL: Eq. (1)/(19), two stochastic views.
        # The paper defines the representation via the final hidden-state [CLS]
        # vector; this also matches the supplied notebook's S_last[:, 0, :].
        mrl_loss, mrl_logits = self.mrl_loss_fn(
            mrl_sentence_embedding(first_output),
            mrl_sentence_embedding(second_output),
        )

        # 2. Official-style alignment execution: compute SIA/PIC on both
        # stochastic views and average the resulting losses. This preserves
        # the implementation pattern of the supplied notebook while using
        # the paper-faithful SIA/PIC formulas inside those modules.
        sia_out_1 = self.sia_loss_fn(
            first_output.hidden_states,
            first_attention_mask,
            special_tokens_mask=first_special_tokens_mask,
        )
        sia_out_2 = self.sia_loss_fn(
            second_output.hidden_states,
            second_attention_mask,
            special_tokens_mask=second_special_tokens_mask,
        )

        sia_loss = (sia_out_1["sia_loss"] + sia_out_2["sia_loss"]) / 2.0
        sia_attn_loss = (
            sia_out_1["attn_loss"] + sia_out_2["attn_loss"]
        ) / 2.0
        sia_cka_loss = (
            sia_out_1["cka_loss"] + sia_out_2["cka_loss"]
        ) / 2.0

        pic_out_1 = self.pic_loss_fn(
            list(first_output.hidden_states),
            first_attention_mask,
        )
        pic_out_2 = self.pic_loss_fn(
            list(second_output.hidden_states),
            second_attention_mask,
        )

        pic_loss = (pic_out_1["total_loss"] + pic_out_2["total_loss"]) / 2.0

        pic_pair_losses: Dict[str, torch.Tensor] = {}
        pair_keys = set(pic_out_1["pair_losses"]) | set(pic_out_2["pair_losses"])
        for key in pair_keys:
            v1 = pic_out_1["pair_losses"].get(key)
            v2 = pic_out_2["pair_losses"].get(key)
            if v1 is not None and v2 is not None:
                pic_pair_losses[key] = (v1 + v2) / 2.0
            elif v1 is not None:
                pic_pair_losses[key] = v1
            else:
                pic_pair_losses[key] = v2

        # Unweighted component sum is retained for ablation/backward checks.
        total_loss = mrl_loss + sia_loss + pic_loss
        mrl_sia_loss = 0.4 * mrl_loss + 0.6 * sia_loss

        return {
            "mrl_loss": mrl_loss,
            "sia_loss": sia_loss,
            "pic_loss": pic_loss,
            "pic_pair_losses": pic_pair_losses,
            "mrl_sia_loss": mrl_sia_loss,
            "total_loss": total_loss,
            "sia_attn_loss": sia_attn_loss,
            "sia_cka_loss": sia_cka_loss,
            "mrl_logits": mrl_logits,
            "pic_out_1": pic_out_1,
            "pic_out_2": pic_out_2,
            "sia_out_1": sia_out_1,
            "sia_out_2": sia_out_2,
            "first_output": first_output,
            "second_output": second_output,
        }



def create_mrl_sia_pic_optimizer(
    model: TinyBERT_MRL_SIA_PIC,
    lr: Optional[float] = None,
    sia_lr_multiplier: Optional[float] = None,
    pic_lr_multiplier: Optional[float] = None,
    weight_decay: Optional[float] = None,
) -> torch.optim.AdamW:
    """Create the AdamW optimizer with parameter groups for TinyBERT, SIA,
    and PIC.
    """

    learning_rate = (
        lr if lr is not None else model.config.learning_rate
    )
    sia_multiplier = (
        sia_lr_multiplier
        if sia_lr_multiplier is not None
        else model.config.sia_lr_multiplier
    )
    pic_multiplier = (
        pic_lr_multiplier
        if pic_lr_multiplier is not None
        else model.config.pic_lr_multiplier
    )
    decay = (
        weight_decay
        if weight_decay is not None
        else model.config.weight_decay
    )

    param_groups = [
        {
            "params": list(model.encoder.parameters()),
            "lr": learning_rate,
            "weight_decay": decay,
        },
        {
            "params": list(model.sia_loss_fn.parameters()),
            "lr": learning_rate * sia_multiplier,
            "weight_decay": decay,
        },
        {
            "params": list(model.pic_loss_fn.parameters()),
            "lr": learning_rate * pic_multiplier,
            "weight_decay": decay,
        },
    ]

    return torch.optim.AdamW(param_groups)


def verify_mrl_sia_pic_optimizer(
    optimizer: torch.optim.Optimizer,
    model: TinyBERT_MRL_SIA_PIC,
) -> bool:
    """Verify that optimizer includes TinyBERT, SIA, and PIC parameters
    without omissions.
    """

    opt_params = {
        p
        for group in optimizer.param_groups
        for p in group["params"]
    }

    encoder_params = set(model.encoder.parameters())
    sia_params = set(model.sia_loss_fn.parameters())
    pic_params = set(model.pic_loss_fn.parameters())

    assert encoder_params.issubset(
        opt_params
    ), "Encoder parameters missing from optimizer."

    assert sia_params.issubset(
        opt_params
    ), "SIA parameters missing from optimizer."

    assert pic_params.issubset(
        opt_params
    ), "PIC parameters missing from optimizer."

    expected_params = encoder_params | sia_params | pic_params

    assert opt_params == expected_params, (
        f"Optimizer parameter mismatch: {len(opt_params)} present vs "
        f"{len(expected_params)} expected."
    )

    assert len(pic_params) > 0, "No PIC parameters found in model."

    return True