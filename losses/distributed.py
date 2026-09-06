"""Small differentiable helpers for distributed contrastive losses."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def gather_with_grad(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """All-gather equal-shaped tensors while preserving autograd."""
    if not is_distributed():
        return (x,)
    return tuple(all_gather(x))


def global_batch_labels(local_batch_size: int, device: torch.device) -> torch.Tensor:
    """Return diagonal positive labels for a rank-local query batch."""
    if is_distributed():
        rank = dist.get_rank()
    else:
        rank = 0
    offset = rank * local_batch_size
    return torch.arange(
        offset,
        offset + local_batch_size,
        device=device,
        dtype=torch.long,
    )
