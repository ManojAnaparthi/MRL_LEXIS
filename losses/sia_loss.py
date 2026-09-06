"""Paper-aligned Semantic Information Alignment (SIA) loss for MIPIC.

Reference equations: MIPIC paper Sec. 3.2 / Appendix A.5.

Key implementation details:
* Teacher attention uses the raw full-dimensional [CLS] token as a fixed query
  anchor and the full hidden states as keys.
* Each lower-dimensional width d_i owns one learnable projection P_i^T:
  R^{d_i} -> R^D, shared across all aligned transformer layers.
* Student attention uses the same full-dimensional [CLS] query anchor.
* Attention matching is KL(student || teacher).
* CKA is evaluated on teacher-ranked top-k tokens.
* The attention/top-k token pool is restricted to *contextual* tokens. Eq. (4)
  defines the teacher over ``h_CLS`` (the query anchor) and ``h_j``, "the j-th
  contextual token"; [CLS]/[SEP]/[PAD] are therefore not keys. Including [CLS]
  as its own key makes s_0 = ||h_CLS||^2 / sqrt(D) dominate every other score
  and collapses a_D onto a one-hot distribution over [CLS], which destroys both
  the top-k importance ranking and the scale of L_att.
* The six SIA prefixes are [16, 32, 64, 128, 256, 512].
* The CKA/attention terms are AVERAGED over dimensions and layers.

  This is a deliberate, approved deviation from Eq. (13)-(14), which write sums.
  With 6 prefixes x 6 layers = 36 terms, summing makes (1-alpha)*(L_SIA+L_PIC)
  24-92x alpha*L_MRL, so alpha=0.4 in Eq. (18) is inoperative and the backbone
  is trained by SIA+PIC alone. The official implementation notebook averages
  (`total_att_loss / att_count`, `total_cka_loss / cka_count`), and the
  best-scoring run of this repository averaged too, logging sia_loss ~0.077
  against ~7.7 for the summed form. Measured on 2xT4, TweetEval macro-F1 at
  d=768: averaged 71.11, summed 51.27.

The implementation intentionally uses float32 for CKA on CUDA. The paper does
not require float64, and float64 is disproportionately slow on NVIDIA T4 GPUs.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_TOPK_RATIOS: Mapping[int, float] = {
    16: 0.2,
    32: 0.3,
    64: 0.4,
    128: 0.5,
    256: 0.6,
    512: 0.7,
}


@torch.jit.ignore
def linear_cka(X: Tensor, Y: Tensor, eps: float = 1e-8, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Return mean-centered linear CKA for 2D/3D tensors, stably in fp32.

    The inputs are normalized before forming Gram matrices. This is algebraically
    equivalent to linear CKA because positive scaling cancels in the normalized
    Frobenius inner product, but it prevents overflow/underflow on low-end GPUs.
    """
    if X.ndim not in (2, 3) or Y.ndim != X.ndim:
        raise ValueError(f"X and Y must both be 2D or 3D tensors; got {X.shape} and {Y.shape}.")
    if X.shape[:-1] != Y.shape[:-1]:
        raise ValueError(f"X and Y must share sample/token axes; got {X.shape} and {Y.shape}.")

    if X.ndim == 2:
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)

    # float64 is more accurate but disproportionately slow on T4; float32 is the
    # default. Switchable so a prior configuration can be reproduced exactly.
    if dtype is None:
        dtype = torch.float32
    if X.device.type == "mps":
        dtype = torch.float32  # MPS has no float64
    X = X.to(dtype)
    Y = Y.to(dtype)
    X = X - X.mean(dim=1, keepdim=True)
    Y = Y - Y.mean(dim=1, keepdim=True)

    x_norm = torch.linalg.vector_norm(X, dim=(-2, -1), keepdim=True).clamp_min(eps)
    y_norm = torch.linalg.vector_norm(Y, dim=(-2, -1), keepdim=True).clamp_min(eps)
    X = X / x_norm
    Y = Y / y_norm

    K = torch.bmm(X, X.transpose(1, 2))
    L = torch.bmm(Y, Y.transpose(1, 2))
    numerator = (K * L).sum(dim=(-2, -1))
    denom = torch.sqrt(
        K.square().sum(dim=(-2, -1)).clamp_min(eps)
        * L.square().sum(dim=(-2, -1)).clamp_min(eps)
    ).clamp_min(eps)
    cka = numerator / denom
    return cka.mean().clamp(0.0, 1.0).float()


def contextual_token_mask(
    attention_mask: Optional[Tensor],
    special_tokens_mask: Optional[Tensor] = None,
) -> Optional[Tensor]:
    """Return the Eq. (4) key mask: real tokens that are not special tokens.

    Rows that would otherwise become empty (a sequence made entirely of special
    tokens) fall back to ``attention_mask`` so the softmax always has support.
    """
    if attention_mask is None:
        return None
    mask = attention_mask.to(torch.long)
    if special_tokens_mask is None:
        return mask
    contextual = mask * (1 - special_tokens_mask.to(torch.long))
    empty = contextual.sum(dim=1, keepdim=True) == 0
    if bool(empty.any()):
        contextual = torch.where(empty, mask, contextual)
    return contextual


class TokenImportanceAttention(nn.Module):
    """Attention helper implementing one paper SIA projection P_i^T."""

    def __init__(self, d_small: int, d_full: int, d_att: int = 64) -> None:
        super().__init__()
        if d_small <= 0 or d_full <= 0:
            raise ValueError("d_small and d_full must be positive.")
        self.d_small = int(d_small)
        self.d_full = int(d_full)
        # d_att is retained only for compatibility with prior constructors.
        self.d_att = int(d_att)
        self.proj_small = nn.Linear(d_small, d_full, bias=False)

    @staticmethod
    def _apply_mask(scores: Tensor, mask: Optional[Tensor]) -> Tensor:
        if mask is None:
            return scores
        return scores.masked_fill(mask == 0, float("-inf"))

    def teacher_scores(
        self,
        full_hidden: Tensor,
        mask: Optional[Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        """Return teacher attention probabilities and logits."""
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        # SIA is deliberately evaluated outside CUDA autocast. The paper uses
        # tau=0.05, and the dot products can overflow when hidden states are
        # fp16. Keeping the attention logits/probabilities in fp32 prevents
        # spurious inf/nan values before the KL term is formed.
        with torch.autocast(device_type="cuda", enabled=False):
            h_cls = full_hidden[:, 0, :].detach().float()
            keys = full_hidden.detach().float()
            scores = torch.bmm(h_cls.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
            scores = scores / math.sqrt(self.d_full)
            scores = self._apply_mask(scores, mask)
            probs = F.softmax(scores / temperature, dim=-1)
        return probs, scores

    def student_scores(
        self,
        small_hidden: Tensor,
        query_cls: Tensor,
        mask: Optional[Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        """Return student attention probabilities and logits."""
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        with torch.autocast(device_type="cuda", enabled=False):
            # Keep projection + attention in fp32 as a numerical-stability
            # boundary around the low-temperature SIA objective.
            projected = self.proj_small(small_hidden.float())
            query = query_cls.detach().float()
            scores = torch.bmm(query.unsqueeze(1), projected.transpose(1, 2)).squeeze(1)
            scores = scores / math.sqrt(self.d_full)
            scores = self._apply_mask(scores, mask)
            probs = F.softmax(scores / temperature, dim=-1)
        return probs, scores

    def token_scores(
        self,
        hidden: Tensor,
        proj: Optional[nn.Linear] = None,
        W_Q: Optional[nn.Linear] = None,
        W_K: Optional[nn.Linear] = None,
        mask: Optional[Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        """Backward-compatible helper for earlier repository callers."""
        del proj, W_Q, W_K
        if hidden.shape[-1] == self.d_full:
            return self.teacher_scores(hidden, mask, temperature)
        query_cls = self.proj_small(hidden[:, 0, :])
        return self.student_scores(hidden, query_cls, mask, temperature)


def attention_kl(student_probs: Tensor, teacher_probs: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute KL(student || teacher), i.e. KL(a_i || a_D), stably in fp32."""
    del eps  # retained for API compatibility
    student = student_probs.float().clamp_min(1e-30)
    teacher = teacher_probs.float().clamp_min(1e-30)
    student = student / student.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    value = (student * (student.log() - teacher.log())).sum(dim=-1).mean()
    return value.clamp_min(0.0)


def _topk_indices(
    scores: Tensor,
    k: int,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Return teacher-ranked token indices, excluding padding."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    if mask is not None:
        valid_counts = mask.to(torch.long).sum(dim=1)
        effective_k = min(k, int(valid_counts.min().item()))
        if effective_k <= 0:
            return torch.empty(
                scores.shape[0], 0, dtype=torch.long, device=scores.device
            )
        scores = scores.masked_fill(mask == 0, float("-inf"))
    else:
        effective_k = min(k, scores.shape[1])

    return torch.topk(scores, k=effective_k, dim=1).indices


def submatrix_cka(
    h_small: Tensor,
    h_full: Tensor,
    scores: Tensor,
    k: int,
    mask: Optional[Tensor] = None,
    eps: float = 1e-8,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Return 1 - linear CKA on teacher-ranked top-k tokens."""
    indices = _topk_indices(scores, k, mask)
    if indices.shape[1] == 0:
        return h_small.new_zeros(())

    small_indices = indices.unsqueeze(-1).expand(-1, -1, h_small.shape[-1])
    full_indices = indices.unsqueeze(-1).expand(-1, -1, h_full.shape[-1])
    selected_small = torch.gather(h_small, 1, small_indices)
    selected_full = torch.gather(h_full, 1, full_indices)
    return 1.0 - linear_cka(selected_small, selected_full, eps=eps, dtype=dtype)


KMap = Union[Dict[int, int], Callable[[int], int]]


class SIALoss(nn.Module):
    """Horizontal SIA objective across layers and lower-dimensional prefixes."""

    def __init__(
        self,
        d_full: int,
        matryoshka_dims: Sequence[int],
        align_layers: Sequence[int],
        d_att: int = 64,
        base_k: int = 64,
        k_map: Optional[KMap] = None,
        topk_ratios: Optional[Mapping[int, float]] = None,
        temperature: float = 0.05,
        w_attn: float = 1.0,
        w_cka: float = 1.0,
        use_attention_kl: bool = True,
        cka_eps: float = 1e-8,
        cka_dtype: str = "float32",
    ) -> None:
        super().__init__()
        if d_full <= 0 or d_att <= 0 or base_k <= 0:
            raise ValueError("d_full, d_att, and base_k must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.d_full = int(d_full)
        self.align_layers = [int(layer) for layer in align_layers]
        self.matryoshka_dims = sorted(
            {int(d) for d in matryoshka_dims if 0 < d < d_full}, reverse=True
        )
        if not self.align_layers:
            raise ValueError("align_layers must contain at least one hidden-state index.")
        if not self.matryoshka_dims:
            raise ValueError("at least one lower-dimensional prefix is required.")

        self.d_att = int(d_att)
        self.base_k = int(base_k)
        self.temperature = float(temperature)
        self.w_attn = float(w_attn)
        self.w_cka = float(w_cka)
        self.use_attention_kl = bool(use_attention_kl)
        self.cka_eps = float(cka_eps)
        if cka_dtype not in {"float32", "float64"}:
            raise ValueError(f"cka_dtype must be float32 or float64, got {cka_dtype!r}.")
        self.cka_dtype = torch.float64 if cka_dtype == "float64" else torch.float32

        ratios = dict(DEFAULT_TOPK_RATIOS if topk_ratios is None else topk_ratios)
        for dim, ratio in ratios.items():
            if ratio <= 0:
                raise ValueError(f"top-k ratio for dim {dim} must be positive.")
        self.topk_ratios = ratios
        self._custom_k_map = k_map is not None

        # One P_i per dimension, shared across all layers, as in Eq. (5).
        # The ModuleDict name remains compatible with the previous test surface.
        self.attn_modules = nn.ModuleDict({
            f"dim_{dim}": TokenImportanceAttention(dim, d_full, d_att)
            for dim in self.matryoshka_dims
        })

        if k_map is None:
            self.k_map: Callable[[int, int], int] = lambda d, m: max(
                8, math.ceil(self.topk_ratios.get(d, d / d_full) * m)
            )
        elif callable(k_map):
            self.k_map = lambda d, m: int(k_map(d))
        elif isinstance(k_map, dict):
            self.k_map = lambda d, m: int(k_map.get(
                d, max(8, math.ceil(self.topk_ratios.get(d, d / d_full) * m))
            ))
        else:
            raise ValueError("k_map must be None, a callable, or a dictionary.")

    def extra_repr(self) -> str:
        return (
            f"d_full={self.d_full}, dims={self.matryoshka_dims}, layers={self.align_layers}, "
            f"topk_ratios={dict(self.topk_ratios)}, temperature={self.temperature}, "
            f"shared_projections=True"
        )

    def _validate_inputs(
        self,
        hidden_states: Sequence[Tensor],
        attention_mask: Optional[Tensor],
    ) -> None:
        if not isinstance(hidden_states, (list, tuple)) or not hidden_states:
            raise ValueError("hidden_states must be a non-empty sequence.")
        max_layer = max(self.align_layers)
        if min(self.align_layers) < 0 or max_layer >= len(hidden_states):
            raise ValueError(
                f"align_layers={self.align_layers} invalid for {len(hidden_states)} hidden states."
            )
        ref_shape = hidden_states[0].shape[:2]
        for index, state in enumerate(hidden_states):
            if state.ndim != 3 or state.shape[-1] != self.d_full:
                raise ValueError(
                    f"hidden_states[{index}] must be [B, L, {self.d_full}], got {tuple(state.shape)}."
                )
            if state.shape[:2] != ref_shape:
                raise ValueError("all hidden_states must share batch and sequence dimensions.")
        if attention_mask is not None:
            if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(ref_shape):
                raise ValueError(
                    f"attention_mask must have shape {tuple(ref_shape)}, got {tuple(attention_mask.shape)}."
                )
            if (attention_mask.sum(dim=1) == 0).any():
                raise ValueError("each attention_mask row must contain at least one valid token.")

    def _topk_count(self, dim: int, sequence_length: int) -> int:
        return int(self.k_map(dim, sequence_length))

    def forward(
        self,
        hidden_states: Sequence[Tensor],
        attention_mask: Optional[Tensor] = None,
        return_breakdown: bool = False,
        special_tokens_mask: Optional[Tensor] = None,
    ) -> Dict[str, Union[Tensor, int, Dict[str, Dict[str, Tensor]]]]:
        self._validate_inputs(hidden_states, attention_mask)
        if special_tokens_mask is not None:
            self._validate_inputs(hidden_states, special_tokens_mask.new_ones(special_tokens_mask.shape))

        # Eq. (4)/(5) range over contextual tokens h_j; [CLS] is the query
        # anchor and [SEP]/[PAD] are not sequence content.
        token_mask = contextual_token_mask(attention_mask, special_tokens_mask)
        # Paper A.5: k_i = max(8, ceil(gamma_i * m)) with m the sequence length,
        # not the padded batch width.
        if token_mask is not None:
            sequence_length = int(token_mask.sum(dim=1).max().item())
        else:
            sequence_length = int(hidden_states[0].shape[1])

        attn_terms = []
        cka_terms = []
        per_term: Dict[str, Dict[str, Tensor]] = {}

        for layer in self.align_layers:
            full = hidden_states[layer]
            # Any one helper can compute the teacher distribution; all use the
            # same full hidden states and the same fixed CLS query.
            reference_module = self.attn_modules[f"dim_{self.matryoshka_dims[0]}"]
            teacher_probs, teacher_scores = reference_module.teacher_scores(
                full, token_mask, self.temperature
            )
            query_cls = full[:, 0, :].detach()

            for dim in self.matryoshka_dims:
                module = self.attn_modules[f"dim_{dim}"]
                small = full[..., :dim]
                student_probs, _ = module.student_scores(
                    small, query_cls, token_mask, self.temperature
                )

                attn_term = (
                    attention_kl(student_probs, teacher_probs)
                    if self.use_attention_kl
                    else full.new_zeros(())
                )
                k = self._topk_count(dim, sequence_length)
                cka_term = submatrix_cka(
                    small,
                    full.detach(),  # Full-dimensional CKA teacher is fixed for this step.
                    teacher_scores,
                    k,
                    token_mask,
                    self.cka_eps,
                    self.cka_dtype,
                )

                attn_terms.append(attn_term)
                cka_terms.append(cka_term)
                if return_breakdown:
                    per_term[f"layer_{layer}_dim_{dim}"] = {
                        "attn_loss": attn_term,
                        "cka_loss": cka_term,
                    }

        # Mean, not sum, over the 36 (layer, dim) terms. See the module
        # docstring: the paper writes sums, but summing leaves Eq. (18)'s alpha
        # inoperative. The official notebook averages; so does the
        # best-scoring run of this repository.
        attn_loss = torch.stack(attn_terms).mean()
        cka_loss = torch.stack(cka_terms).mean()
        sia_loss = self.w_attn * attn_loss + self.w_cka * cka_loss

        result: Dict[str, Union[Tensor, int, Dict[str, Dict[str, Tensor]]]] = {
            "sia_loss": sia_loss,
            "attn_loss": attn_loss,
            "cka_loss": cka_loss,
            "n_terms": len(attn_terms),
        }
        if return_breakdown:
            result["per_term"] = per_term
        return result
