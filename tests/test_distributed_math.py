"""Minimal 2-rank CPU test for the DDP negative-gather contract."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from losses.distributed import gather_with_grad, global_batch_labels


def _worker(rank: int, world_size: int, init_file: str, results_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    x = torch.tensor([[rank + 1.0], [rank + 2.0]], requires_grad=True)
    gathered = torch.cat(gather_with_grad(x), dim=0)
    expected = torch.tensor([[1.0], [2.0], [2.0], [3.0]])
    assert torch.equal(gathered.detach(), expected)
    labels = global_batch_labels(2, x.device)
    assert labels.tolist() == [rank * 2, rank * 2 + 1]

    # The autograd-preserving all_gather must route a gradient from the
    # gathered representation back to the rank-local tensor.
    loss = gathered.sum()
    loss.backward()
    grad = x.grad.detach().cpu()
    assert torch.equal(grad, torch.full_like(grad, float(world_size)))

    Path(results_file).write_text("ok", encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


def test_two_rank_gather_preserves_grad_and_labels(tmp_path: Path) -> None:
    init_file = str(tmp_path / "init")
    results = [str(tmp_path / f"rank{rank}.txt") for rank in range(2)]
    mp.spawn(_worker, args=(2, init_file, results[0]), nprocs=2, join=True)
    assert Path(results[0]).exists()
