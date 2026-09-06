"""Real two-rank DDP test for the MIPIC loss graph (gloo/CPU, no GPU needed).

This is the test that settles the ``find_unused_parameters`` question from the
audit brief: it wraps the actual composite module in ``DistributedDataParallel``
with ``find_unused_parameters=False`` and runs two steps. If any parameter in
the reducer failed to receive a gradient, DDP raises on the second forward.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


HIDDEN = 32
LAYERS = 6
SEQ = 12


class _StubEncoder(nn.Module):
    """A 6-layer stand-in with the same output contract as TinyBERT-6L.

    It carries a ``pooler`` so the frozen-pooler behaviour is exercised, and
    returns ``LAYERS + 1`` hidden states like ``BertModel``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(64, HIDDEN)
        self.layers = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN) for _ in range(LAYERS))
        self.pooler = nn.Module()
        self.pooler.dense = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True, return_dict=True):
        del attention_mask, output_hidden_states, return_dict
        state = self.embeddings(input_ids)
        # Dropout gives the two views genuinely different activations.
        state = torch.nn.functional.dropout(state, p=0.1, training=self.training)
        states = [state]
        for layer in self.layers:
            state = torch.tanh(layer(state))
            state = torch.nn.functional.dropout(state, p=0.1, training=self.training)
            states.append(state)

        class _Out:
            pass

        out = _Out()
        out.hidden_states = tuple(states)
        out.last_hidden_state = states[-1]
        out.pooler_output = torch.tanh(self.pooler.dense(states[-1][:, 0, :]))
        return out


def _make_batch(rank: int, batch_size: int = 4):
    generator = torch.Generator().manual_seed(100 + rank)
    ids = torch.randint(0, 64, (batch_size, SEQ), generator=generator)
    attention = torch.ones(batch_size, SEQ, dtype=torch.long)
    special = torch.zeros(batch_size, SEQ, dtype=torch.long)
    special[:, 0] = 1
    special[:, -1] = 1
    return {
        "input_ids1_stu": ids,
        "attention_mask1_stu": attention,
        "special_tokens_mask1_stu": special,
        "input_ids2_stu": ids.clone(),
        "attention_mask2_stu": attention.clone(),
        "special_tokens_mask2_stu": special.clone(),
    }


def _worker(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.mipic import MIPICConfig, TinyBERT_MIPIC
    from models.mrl_sia_pic import MRLSIAPICConfig, TinyBERT_MRL_SIA_PIC

    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    torch.manual_seed(0)

    encoder = _StubEncoder()
    core = TinyBERT_MRL_SIA_PIC.__new__(TinyBERT_MRL_SIA_PIC)
    nn.Module.__init__(core)
    core.config = MRLSIAPICConfig(
        matryoshka_dims=(4, 8, 16, HIDDEN),
        align_layers=(1, 3, 6),
        pic_pipeline_pairs=((1, 4, 3, 8, 6, HIDDEN),),
    )
    core.encoder = encoder
    from losses.mrl_loss import MatryoshkaInfoNCELoss
    from losses.pic_loss import PICModule
    from losses.sia_loss import SIALoss

    core.mrl_loss_fn = MatryoshkaInfoNCELoss(0.05, core.config.matryoshka_dims)
    core.sia_loss_fn = SIALoss(
        d_full=HIDDEN,
        matryoshka_dims=core.config.matryoshka_dims,
        align_layers=core.config.align_layers,
        temperature=core.config.sia_temperature,
        topk_ratios={4: 0.2, 8: 0.4, 16: 0.6},
    )
    core.pic_loss_fn = PICModule(
        d_full=HIDDEN,
        pipeline_pairs=[list(p) for p in core.config.pic_pipeline_pairs],
        temperature=0.05,
    )
    model = TinyBERT_MIPIC(underlying_model=core, config=MIPICConfig())

    # Patch the hidden-state count check for the stub's LAYERS + 1 outputs.
    ddp = DDP(model, find_unused_parameters=False, broadcast_buffers=False)
    optimizer = torch.optim.AdamW([p for p in ddp.parameters() if p.requires_grad], lr=1e-3)

    trainable = [(n, p) for n, p in ddp.named_parameters() if p.requires_grad]
    frozen = [n for n, p in ddp.named_parameters() if not p.requires_grad]

    losses = []
    for step in range(2):
        batch = _make_batch(rank * 10 + step)
        optimizer.zero_grad(set_to_none=True)
        out = ddp(batch=batch)
        out["loss"].backward()
        missing = [n for n, p in trainable if p.grad is None]
        assert not missing, f"rank {rank}: no gradient for {missing}"
        optimizer.step()
        losses.append(float(out["loss"].detach()))

    # After DDP all-reduce every rank must hold identical parameters.
    signature = torch.cat([p.detach().reshape(-1) for _, p in trainable])
    gathered = [torch.zeros_like(signature) for _ in range(world_size)]
    dist.all_gather(gathered, signature)
    assert torch.allclose(gathered[0], gathered[1], atol=0, rtol=0), (
        "DDP gradient synchronisation left the ranks with different weights"
    )

    Path(out_dir, f"rank{rank}.txt").write_text(
        f"{len(trainable)} trainable; frozen={sorted(frozen)}; losses={losses}",
        encoding="utf-8",
    )
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_two_rank_mipic_needs_no_unused_parameter_detection(tmp_path: Path) -> None:
    """The single shipped configuration must run under find_unused_parameters=False.

    z is pooler_output, so the pooler is consumed by L_MRL and receives a
    gradient like everything else. Nothing is frozen, and the worker asserts
    every parameter has a gradient before each optimizer step -- which is the
    real check; DDP would also raise on the second forward if one did not.
    """
    init_file = str(tmp_path / "init")
    mp.spawn(_worker, args=(2, init_file, str(tmp_path)), nprocs=2, join=True)

    reports = [(tmp_path / f"rank{rank}.txt").read_text(encoding="utf-8") for rank in range(2)]
    assert len(reports) == 2
    for report in reports:
        assert "frozen=[]" in report, f"nothing should be frozen: {report}"
