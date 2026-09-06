"""MRL loss reused from ``mrl-baseline-a1.ipynb``."""

from typing import Dict, Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from losses.distributed import gather_with_grad, global_batch_labels, is_distributed


MRL_NESTED_DIMS = (16, 32, 64, 128, 256, 512, 768)
MRL_TEMPERATURE = 0.05


def info_nce(
    q: torch.Tensor,
    k: torch.Tensor,
    temperature: float = MRL_TEMPERATURE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Notebook SimCSE-style InfoNCE helper retained for baseline logging."""
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = torch.matmul(q, k.T) / temperature
    labels = torch.arange(q.size(0), device=q.device)
    loss_inbatch = F.cross_entropy(logits, labels)
    return loss_inbatch, logits


def Matry_infonce(
    a: torch.Tensor,
    b: torch.Tensor,
    temperature: float = MRL_TEMPERATURE,
    nested_dims: Iterable[int] = MRL_NESTED_DIMS,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Sum InfoNCE losses over valid nested prefixes of paired CLS embeddings.

    This is the active ``Matry_infonce`` implementation from the uploaded
    baseline notebook, separated from notebook cells without changing its
    normalization, logits, labels, or unnormalised sum across dimensions.
    """
    assert a.dim() == 2, f"Expected a to be 2D [batch, dim], got shape {a.shape}"
    assert b.dim() == 2, f"Expected b to be 2D [batch, dim], got shape {b.shape}"
    assert a.shape == b.shape, f"a and b must have same shape, got {a.shape} vs {b.shape}"

    # Loss arithmetic stays in fp32 even when the encoder runs under CUDA AMP.
    a = a.float()
    b = b.float()
    total_loss = 0.0
    all_logits: Dict[str, torch.Tensor] = {}
    full_dim = a.size(1)

    for dim in nested_dims:
        if dim > full_dim:
            print(f"Warning: skipping dim={dim} as it exceeds full_dim={full_dim}")
            continue

        q = F.normalize(a[:, :dim], dim=-1)
        k = F.normalize(b[:, :dim], dim=-1)

        # Under DDP, keep the local query batch but expose all ranks' keys so
        # the InfoNCE denominator uses the global effective batch size.
        gathered_keys = torch.cat(gather_with_grad(k), dim=0)
        logits = torch.matmul(q, gathered_keys.T) / temperature
        labels = global_batch_labels(q.size(0), q.device)
        total_loss += F.cross_entropy(logits, labels)
        all_logits[f"dim_{dim}"] = logits

    return total_loss, all_logits


class MatryoshkaInfoNCELoss(nn.Module):
    """Configurable module wrapper around the notebook's ``Matry_infonce``."""

    def __init__(
        self,
        temperature: float = MRL_TEMPERATURE,
        nested_dims: Iterable[int] = MRL_NESTED_DIMS,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.nested_dims = tuple(nested_dims)

    def forward(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return Matry_infonce(first, second, self.temperature, self.nested_dims)
