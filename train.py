"""Callable refactor of the active MRL baseline notebook training path.

This module intentionally trains only MRL. It contains no SIA, PIC, or MIPIC
loss composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from losses.mrl_loss import MRL_NESTED_DIMS, MRL_TEMPERATURE, MatryoshkaInfoNCELoss, info_nce
from models.mipic import TinyBERT_MIPIC, create_mipic_optimizer
from models.mrl_pic import TinyBERT_MRL_PIC, create_mrl_pic_optimizer
from models.mrl_sia import TinyBERT_MRL_SIA, create_mrl_sia_optimizer
from models.mrl_sia_pic import TinyBERT_MRL_SIA_PIC, create_mrl_sia_pic_optimizer
from models.tinybert import mrl_sentence_embedding


# The exact ``source`` values present in the supplied final_data.csv pool.
# The paper's Appendix A.2 accounting (24,000 sentences from Banking77,
# TweetEval, STS-B, SICK, MRPC, WiC) does not reconcile with this 20,240-row
# pool and never names DGEM_FORMAT. The discrepancy is reported, not silently
# corrected: nothing here filters the corpus, and DGEM_FORMAT is retained.
PUBLISHED_TRAINING_SOURCES: Tuple[str, ...] = (
    "BANKING77",
    "TWEET_EVAL_SENT",
    "MRPC",
    "DEEHAN1866/WIC",
    "DGEM_FORMAT",
    "SICK",
    "STSB",
)

def _amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda" and os.environ.get("MIPIC_AMP", "1").lower() not in {"0", "false", "no"}


def _make_scaler(enabled: bool):
    if not enabled:
        try:
            return torch.amp.GradScaler("cuda", enabled=False)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=False)
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _autocast(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


_LAST_GRAD_NORM: Dict[str, float] = {"value": float("nan")}


def _clip_gradients(model, optimizer, scaler, amp_enabled: bool, max_grad_norm: float):
    """Measure the gradient norm, and clip only if a positive bound is set.

    The norm is ALWAYS recorded (it is nearly free once gradients exist) and
    printed with each heartbeat, so a run reports whether clipping would have
    bound rather than leaving it to guesswork.

    max_grad_norm defaults to 0.0 = no clipping, matching the reference MRL
    baseline whose TweetEval score is neutral against the untrained backbone
    (71.61 vs 71.99 at d=768). Our clipped 2xT4 run lost 4.75 F1 over the same
    5 epochs. The official notebook does clip at 1.0, but only around the much
    larger SIA/PIC alignment gradients.
    """
    if amp_enabled:
        scaler.unscale_(optimizer)
    bound = max_grad_norm if (max_grad_norm and max_grad_norm > 0) else float("inf")
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=bound))
    _LAST_GRAD_NORM["value"] = norm
    return norm


def _optimizer_step(optimizer, scaler, amp_enabled: bool, scheduler=None):
    """Take one optimizer step and advance the scheduler only after a real step."""
    if amp_enabled:
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        # A scale reduction means GradScaler found non-finite gradients and skipped the step.
        stepped = new_scale >= old_scale
    else:
        optimizer.step()
        stepped = True
    if scheduler is not None and stepped:
        scheduler.step()
    return stepped


def _max_batches() -> Optional[int]:
    value = os.environ.get("MIPIC_MAX_BATCHES")
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"MIPIC_MAX_BATCHES must be an integer, got {value!r}") from exc
    return parsed if parsed > 0 else None


def _log_every() -> int:
    """Heartbeat interval in batches.

    Defaults to 100, not 500: a 500-batch gap makes a DDP stall look like a
    1265-batch epoch that simply stopped, and gives no way to localise it.
    """
    try:
        return max(1, int(os.environ.get("MIPIC_LOG_EVERY", "100")))
    except ValueError:
        return 100


def _stall_watchdog_seconds() -> float:
    """Seconds of *no training progress* before a rank dumps its own stack."""
    try:
        return float(os.environ.get("MIPIC_STACK_DUMP_SECONDS", "0") or 0)
    except ValueError:
        return 0.0


def _arm_stall_watchdog() -> None:
    """(Re)start the no-progress watchdog. Called on every heartbeat.

    faulthandler.dump_traceback_later resets a pending timer when called again,
    so re-arming it from the heartbeat converts an unconditional periodic dump
    into a true stall detector: it fires only if the interval between two
    heartbeats exceeds the timeout, i.e. only if training actually stopped.
    """
    seconds = _stall_watchdog_seconds()
    if seconds <= 0:
        return
    import faulthandler
    import sys as _sys

    try:
        faulthandler.dump_traceback_later(seconds, repeat=True, exit=False, file=_sys.stderr)
    except (RuntimeError, ValueError):
        pass


def _disarm_stall_watchdog() -> None:
    """Stop the watchdog once the training loop is done.

    Evaluation and checkpointing are not heartbeat-driven, so leaving the
    watchdog armed across them would produce spurious dumps.
    """
    if _stall_watchdog_seconds() <= 0:
        return
    import faulthandler

    try:
        faulthandler.cancel_dump_traceback_later()
    except (RuntimeError, ValueError):
        pass


def _rank_tag() -> str:
    if dist.is_available() and dist.is_initialized():
        return f"[rank {dist.get_rank()}]"
    return "[rank 0]"


def _heartbeat(epoch_index: int, total_epochs: int, batch_index: int, total_batches: int) -> None:
    """Print progress from EVERY rank.

    Rank-0-only logging cannot distinguish "rank 0 is stuck" from "rank 1 is
    stuck and rank 0 is blocked waiting for it in a collective". Both ranks
    printing makes the last surviving line identify the stalled rank.
    """
    import time as _time

    norm = _LAST_GRAD_NORM["value"]
    grad = "" if norm != norm else f"  |g|={norm:.3f}"
    print(
        f"{_rank_tag()} {_time.strftime('%H:%M:%S')} epoch {epoch_index}/{total_epochs}: "
        f"batch {batch_index}/{total_batches}{grad}",
        flush=True,
    )
    _arm_stall_watchdog()


@dataclass(frozen=True)
class MRLBaselineConfig:
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 2e-5
    max_length: int = 256
    temperature: float = MRL_TEMPERATURE
    nested_dims: Tuple[int, ...] = MRL_NESTED_DIMS
    warmup_ratio: float = 0.1
    min_learning_rate: float = 2e-6
    num_workers: int = 2
    # 0.0 = no clipping. See _clip_gradients for the measured justification.
    max_grad_norm: float = 0.0


class TextPairRaw(Dataset[Tuple[str, str]]):
    """The baseline's positive-pair dataset for ``TASK_TYPE = 'pair_cls'``."""

    def __init__(self, texts: Sequence[str]) -> None:
        self.samples = [(str(text), str(text)) for text in texts]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.samples[index]


class PairTokenizerCollate:
    """Emit the student batch keys consumed by the notebook's active loop."""

    def __init__(self, tokenizer: object, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        first, second = zip(*batch)
        first_encoded = self.tokenizer(
            list(first),
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        second_encoded = self.tokenizer(
            list(second),
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        return {
            "input_ids1_stu": first_encoded["input_ids"],
            "attention_mask1_stu": first_encoded["attention_mask"],
            "special_tokens_mask1_stu": first_encoded["special_tokens_mask"],
            "input_ids2_stu": second_encoded["input_ids"],
            "attention_mask2_stu": second_encoded["attention_mask"],
            "special_tokens_mask2_stu": second_encoded["special_tokens_mask"],
        }


def build_mrl_train_loader(
    csv_path: str | Path,
    tokenizer: object,
    config: MRLBaselineConfig,
) -> DataLoader[Dict[str, torch.Tensor]]:
    """Build the notebook's ``text -> (premise, hypothesis)`` positive-pair loader."""
    import pandas as pd

    frame = pd.read_csv(csv_path)
    if "text" not in frame:
        raise ValueError(f"{csv_path} must contain the baseline 'text' column.")

    # The supplied final_data.csv is the training corpus consumed by the
    # official implementation notebook. Keep it unchanged; the paper's
    # dataset accounting is inconsistent with the supplied 20,240-row pool.
    if "source" in frame.columns and _maybe_rank0():
        print(
            f"Training corpus: {len(frame):,} rows; sources="
            f"{frame['source'].astype(str).value_counts().to_dict()}",
            flush=True,
        )
    dataset = TextPairRaw(frame["text"].astype(str).tolist())

    distributed = dist.is_available() and dist.is_initialized()
    sampler = DistributedSampler(
        dataset,
        shuffle=True,
        drop_last=False,
    ) if distributed else None

    if _maybe_rank0():
        world_size = dist.get_world_size() if distributed else 1
        print(
            f"DataLoader: rows={len(dataset):,}, per_rank_batch={config.batch_size}, "
            f"world_size={world_size}, effective_global_batch={config.batch_size * world_size}, "
            f"sampler={'DistributedSampler' if distributed else 'local'}",
            flush=True,
        )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=PairTokenizerCollate(tokenizer, config.max_length),
        pin_memory=torch.cuda.is_available(),
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )


def create_mrl_optimizer_scheduler(
    model: torch.nn.Module,
    train_loader: DataLoader[Dict[str, torch.Tensor]],
    config: MRLBaselineConfig,
) -> Tuple[torch.optim.Optimizer, object]:
    """Create the AdamW and cosine-with-min-lr scheduler used by the notebook."""
    from transformers import get_scheduler

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_scheduler(
        name="cosine_with_min_lr",
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
        scheduler_specific_kwargs={"min_lr": config.min_learning_rate},
    )
    return optimizer, scheduler


def _set_distributed_sampler_epoch(train_loader: DataLoader, epoch_index: int) -> None:
    """Set the epoch on a DistributedSampler so every rank gets a new shuffle."""
    sampler = getattr(train_loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch_index)


def _is_distributed_model(model: torch.nn.Module) -> bool:
    return isinstance(model, DDP)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model for checkpointing/attribute access."""
    return model.module if isinstance(model, DDP) else model


def _distributed_average(value: float, count: int, device: torch.device) -> tuple[float, int]:
    """Average a scalar accumulated over samples across all DDP ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return value, count

    stats = torch.tensor(
        [value, float(count)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item()), int(stats[1].item())


def _maybe_rank0() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _compute_losses(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Compute losses through the model's forward path so DDP hooks run."""
    if isinstance(model, DDP):
        return model(batch=batch)
    return model.compute_losses(batch=batch)


def train_mrl_baseline(
    model: torch.nn.Module,
    train_loader: DataLoader[Dict[str, torch.Tensor]],
    config: MRLBaselineConfig,
    device: torch.device,
    *,
    start_epoch: int = 0,
    epoch_callback: Optional[Callable[[int, torch.nn.Module, torch.optim.Optimizer, object, List[Dict[str, float]]], None]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
) -> List[Dict[str, float]]:
    """Run the notebook's MRL-only training loop when explicitly invoked."""
    model.to(device)
    loss_module = MatryoshkaInfoNCELoss(config.temperature, config.nested_dims)
    if optimizer is None or scheduler is None:
        optimizer, scheduler = create_mrl_optimizer_scheduler(model, train_loader, config)
    amp_enabled = _amp_enabled(device)
    scaler = _make_scaler(amp_enabled)
    max_batches = _max_batches()
    log_every = _log_every()
    history: List[Dict[str, float]] = []

    for epoch_index in range(start_epoch, config.epochs):
        _set_distributed_sampler_epoch(train_loader, epoch_index)
        model.train()
        total_loss = 0.0
        total_task_loss = 0.0
        item_count = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            if batch_index == 1 or batch_index % log_every == 0:
                _heartbeat(epoch_index + 1, config.epochs, batch_index, len(train_loader))
            first_ids = batch["input_ids1_stu"].to(device, non_blocking=True)
            first_mask = batch["attention_mask1_stu"].to(device, non_blocking=True)
            second_ids = batch["input_ids2_stu"].to(device, non_blocking=True)
            second_mask = batch["attention_mask2_stu"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with _autocast(amp_enabled):
                first_output = model(
                    input_ids=first_ids,
                    attention_mask=first_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                second_output = model(
                    input_ids=second_ids,
                    attention_mask=second_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                # Same read-out as every other arm, so the four-way comparison
                # stays controlled. Under the default 'mlp_train_only' this is
                # pooler_output (the SimCSE head, discarded at inference);
                # under 'cls' it is hidden_states[-1][:, 0, :].
                first_view = mrl_sentence_embedding(first_output)
                second_view = mrl_sentence_embedding(second_output)
                task_loss, _ = info_nce(first_view, second_view, config.temperature)
                loss, _ = loss_module(first_view, second_view)

            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch_index + 1}, batch={batch_index}: "
                    f"{loss.detach().float().item()} (mrl_loss={loss.detach().float().item()})"
                )
            scaler.scale(loss.float()).backward()
            _clip_gradients(model, optimizer, scaler, amp_enabled, config.max_grad_norm)
            stepped = _optimizer_step(optimizer, scaler, amp_enabled, scheduler)
            if not stepped and not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Non-finite loss after skipped AMP step at epoch={epoch_index + 1}, batch={batch_index}")

            batch_size = first_ids.size(0)
            total_loss += loss.detach().item() * batch_size
            total_task_loss += task_loss.detach().item() * batch_size
            item_count += batch_size

        total_loss, item_count = _distributed_average(total_loss, item_count, device)
        total_task_loss, _ = _distributed_average(total_task_loss, item_count, device)

        history.append(
            {
                "mrl_loss": total_loss / max(1, item_count),
                "task_loss": total_task_loss / max(1, item_count),
            }
        )
        if epoch_callback is not None:
            epoch_callback(epoch_index + 1, model, optimizer, scheduler, history)

    _disarm_stall_watchdog()
    return history


def save_mrl_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> None:
    """Save exactly the model state dictionary saved by the baseline notebook."""
    torch.save(model.state_dict(), checkpoint_path)


def load_mrl_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load a baseline state-dictionary checkpoint and put the model in eval mode."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def train_mrl_sia(
    model: TinyBERT_MRL_SIA,
    train_loader: DataLoader[Dict[str, torch.Tensor]],
    device: torch.device,
    epochs: int = 1,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    *,
    start_epoch: int = 0,
    epoch_callback: Optional[Callable[[int, torch.nn.Module, torch.optim.Optimizer, object, List[Dict[str, float]]], None]] = None,
    max_grad_norm: float = 0.0,
) -> List[Dict[str, float]]:
    """Run combined MRL + SIA training loop across the provided dataloader."""
    model.to(device)
    if optimizer is None:
        optimizer = create_mrl_sia_optimizer(model)
    amp_enabled = _amp_enabled(device)
    scaler = _make_scaler(amp_enabled)
    max_batches = _max_batches()
    log_every = _log_every()
    history: List[Dict[str, float]] = []

    for epoch_index in range(start_epoch, start_epoch + epochs):
        _set_distributed_sampler_epoch(train_loader, epoch_index)
        model.train()
        total_loss = 0.0
        total_mrl_loss = 0.0
        total_sia_loss = 0.0
        total_sia_attn_loss = 0.0
        total_sia_cka_loss = 0.0
        item_count = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            device_batch = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
                if torch.is_tensor(v)
            }
            if max_batches is not None and batch_index > max_batches:
                break
            if batch_index == 1 or batch_index % log_every == 0:
                _heartbeat(epoch_index + 1, start_epoch + epochs, batch_index, len(train_loader))

            optimizer.zero_grad(set_to_none=True)
            with _autocast(amp_enabled):
                loss_dict = _compute_losses(model, device_batch)
                loss = loss_dict["total_loss"]

            if not torch.isfinite(loss.detach()):
                components = []
                for name in ("mrl_loss", "sia_loss", "sia_attn_loss", "sia_cka_loss", "pic_loss"):
                    value = loss_dict.get(name)
                    if torch.is_tensor(value):
                        item = value.detach().float().item()
                        components.append(f"{name}={item}")
                detail = ", ".join(components)
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch_index + 1}, batch={batch_index}: "
                    f"{loss.detach().float().item()} ({detail})"
                )
            scaler.scale(loss.float()).backward()
            _clip_gradients(model, optimizer, scaler, amp_enabled, max_grad_norm)
            stepped = _optimizer_step(optimizer, scaler, amp_enabled, scheduler)

            batch_size = device_batch["input_ids1_stu"].size(0)
            total_loss += loss.detach().item() * batch_size
            total_mrl_loss += loss_dict["mrl_loss"].detach().item() * batch_size
            total_sia_loss += loss_dict["sia_loss"].detach().item() * batch_size
            total_sia_attn_loss += loss_dict["sia_attn_loss"].detach().item() * batch_size
            total_sia_cka_loss += loss_dict["sia_cka_loss"].detach().item() * batch_size
            item_count += batch_size

        total_loss, global_count = _distributed_average(total_loss, item_count, device)
        total_mrl_loss, _ = _distributed_average(total_mrl_loss, item_count, device)
        total_sia_loss, _ = _distributed_average(total_sia_loss, item_count, device)
        total_sia_attn_loss, _ = _distributed_average(total_sia_attn_loss, item_count, device)
        total_sia_cka_loss, _ = _distributed_average(total_sia_cka_loss, item_count, device)
        item_count = global_count

        history.append(
            {
                "total_loss": total_loss / max(1, item_count),
                "mrl_loss": total_mrl_loss / max(1, item_count),
                "sia_loss": total_sia_loss / max(1, item_count),
                "sia_attn_loss": total_sia_attn_loss / max(1, item_count),
                "sia_cka_loss": total_sia_cka_loss / max(1, item_count),
            }
        )

        if epoch_callback is not None:
            epoch_callback(epoch_index + 1, model, optimizer, scheduler, history)

    _disarm_stall_watchdog()
    return history


def train_mrl_sia_pic(
    model: TinyBERT_MRL_SIA_PIC,
    train_loader: DataLoader[Dict[str, torch.Tensor]],
    device: torch.device,
    epochs: int = 1,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    *,
    start_epoch: int = 0,
    epoch_callback: Optional[Callable[[int, torch.nn.Module, torch.optim.Optimizer, object, List[Dict[str, float]]], None]] = None,
    max_grad_norm: float = 0.0,
) -> List[Dict[str, float]]:
    """Run combined MRL + SIA + PIC training loop across the provided dataloader."""
    model.to(device)
    if optimizer is None:
        optimizer = create_mrl_sia_pic_optimizer(model)
    amp_enabled = _amp_enabled(device)
    scaler = _make_scaler(amp_enabled)
    max_batches = _max_batches()
    log_every = _log_every()
    history: List[Dict[str, float]] = []

    for epoch_index in range(start_epoch, start_epoch + epochs):
        _set_distributed_sampler_epoch(train_loader, epoch_index)
        model.train()
        total_loss = 0.0
        total_mrl_loss = 0.0
        total_sia_loss = 0.0
        total_pic_loss = 0.0
        item_count = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            device_batch = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
                if torch.is_tensor(v)
            }
            if max_batches is not None and batch_index > max_batches:
                break
            if batch_index == 1 or batch_index % log_every == 0:
                _heartbeat(epoch_index + 1, start_epoch + epochs, batch_index, len(train_loader))

            optimizer.zero_grad(set_to_none=True)
            with _autocast(amp_enabled):
                loss_dict = _compute_losses(model, device_batch)
                loss = loss_dict["total_loss"]

            if not torch.isfinite(loss.detach()):
                components = []
                for name in ("mrl_loss", "sia_loss", "sia_attn_loss", "sia_cka_loss", "pic_loss"):
                    value = loss_dict.get(name)
                    if torch.is_tensor(value):
                        item = value.detach().float().item()
                        components.append(f"{name}={item}")
                detail = ", ".join(components)
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch_index + 1}, batch={batch_index}: "
                    f"{loss.detach().float().item()} ({detail})"
                )
            scaler.scale(loss.float()).backward()
            _clip_gradients(model, optimizer, scaler, amp_enabled, max_grad_norm)
            stepped = _optimizer_step(optimizer, scaler, amp_enabled, scheduler)

            batch_size = device_batch["input_ids1_stu"].size(0)
            total_loss += loss.detach().item() * batch_size
            total_mrl_loss += loss_dict["mrl_loss"].detach().item() * batch_size
            total_sia_loss += loss_dict["sia_loss"].detach().item() * batch_size
            total_pic_loss += loss_dict["pic_loss"].detach().item() * batch_size
            item_count += batch_size

        total_loss, global_count = _distributed_average(total_loss, item_count, device)
        total_mrl_loss, _ = _distributed_average(total_mrl_loss, item_count, device)
        total_sia_loss, _ = _distributed_average(total_sia_loss, item_count, device)
        total_pic_loss, _ = _distributed_average(total_pic_loss, item_count, device)
        item_count = global_count

        history.append(
            {
                "total_loss": total_loss / max(1, item_count),
                "mrl_loss": total_mrl_loss / max(1, item_count),
                "sia_loss": total_sia_loss / max(1, item_count),
                "pic_loss": total_pic_loss / max(1, item_count),
            }
        )

        if epoch_callback is not None:
            epoch_callback(epoch_index + 1, model, optimizer, scheduler, history)

    _disarm_stall_watchdog()
    return history


def train_mipic(
    model: TinyBERT_MIPIC,
    train_loader: DataLoader[Dict[str, torch.Tensor]],
    device: torch.device,
    epochs: int = 1,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    *,
    start_epoch: int = 0,
    epoch_callback: Optional[Callable[[int, torch.nn.Module, torch.optim.Optimizer, object, List[Dict[str, float]]], None]] = None,
    max_grad_norm: float = 0.0,
) -> List[Dict[str, float]]:
    """Run complete MIPIC training loop across the provided dataloader."""
    model.to(device)
    if optimizer is None:
        optimizer = create_mipic_optimizer(model)
    amp_enabled = _amp_enabled(device)
    scaler = _make_scaler(amp_enabled)
    max_batches = _max_batches()
    log_every = _log_every()
    history: List[Dict[str, float]] = []

    for epoch_index in range(start_epoch, start_epoch + epochs):
        _set_distributed_sampler_epoch(train_loader, epoch_index)
        model.train()
        total_loss = 0.0
        total_mrl_loss = 0.0
        total_sia_loss = 0.0
        total_pic_loss = 0.0
        item_count = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            device_batch = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
                if torch.is_tensor(v)
            }
            if max_batches is not None and batch_index > max_batches:
                break
            if batch_index == 1 or batch_index % log_every == 0:
                _heartbeat(epoch_index + 1, start_epoch + epochs, batch_index, len(train_loader))

            optimizer.zero_grad(set_to_none=True)
            with _autocast(amp_enabled):
                loss_dict = _compute_losses(model, device_batch)
                loss = loss_dict["loss"]

            if not torch.isfinite(loss.detach()):
                components = []
                for name in ("mrl_loss", "sia_loss", "sia_attn_loss", "sia_cka_loss", "pic_loss"):
                    value = loss_dict.get(name)
                    if torch.is_tensor(value):
                        item = value.detach().float().item()
                        components.append(f"{name}={item}")
                detail = ", ".join(components)
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch_index + 1}, batch={batch_index}: "
                    f"{loss.detach().float().item()} ({detail})"
                )
            scaler.scale(loss.float()).backward()
            _clip_gradients(model, optimizer, scaler, amp_enabled, max_grad_norm)
            stepped = _optimizer_step(optimizer, scaler, amp_enabled, scheduler)

            batch_size = device_batch["input_ids1_stu"].size(0)
            total_loss += loss.detach().item() * batch_size
            total_mrl_loss += loss_dict["mrl_loss"].detach().item() * batch_size
            total_sia_loss += loss_dict["sia_loss"].detach().item() * batch_size
            total_pic_loss += loss_dict["pic_loss"].detach().item() * batch_size
            item_count += batch_size

        total_loss, global_count = _distributed_average(total_loss, item_count, device)
        total_mrl_loss, _ = _distributed_average(total_mrl_loss, item_count, device)
        total_sia_loss, _ = _distributed_average(total_sia_loss, item_count, device)
        total_pic_loss, _ = _distributed_average(total_pic_loss, item_count, device)
        item_count = global_count

        history.append(
            {
                "total_loss": total_loss / max(1, item_count),
                "mipic_loss": total_loss / max(1, item_count),
                "mrl_loss": total_mrl_loss / max(1, item_count),
                "sia_loss": total_sia_loss / max(1, item_count),
                "pic_loss": total_pic_loss / max(1, item_count),
            }
        )

        if epoch_callback is not None:
            epoch_callback(epoch_index + 1, model, optimizer, scheduler, history)

    _disarm_stall_watchdog()
    return history


def train_mrl_pic(
    model: TinyBERT_MRL_PIC,
    train_loader: DataLoader[Dict[str, torch.Tensor]],
    device: torch.device,
    epochs: int = 1,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    *,
    start_epoch: int = 0,
    epoch_callback: Optional[Callable[[int, torch.nn.Module, torch.optim.Optimizer, object, List[Dict[str, float]]], None]] = None,
    max_grad_norm: float = 0.0,
) -> List[Dict[str, float]]:
    """Run MRL + PIC ablation training loop across the provided dataloader."""
    model.to(device)
    if optimizer is None:
        optimizer = create_mrl_pic_optimizer(model)
    amp_enabled = _amp_enabled(device)
    scaler = _make_scaler(amp_enabled)
    max_batches = _max_batches()
    log_every = _log_every()
    history: List[Dict[str, float]] = []

    for epoch_index in range(start_epoch, start_epoch + epochs):
        _set_distributed_sampler_epoch(train_loader, epoch_index)
        model.train()
        total_loss = 0.0
        total_mrl_loss = 0.0
        total_pic_loss = 0.0
        item_count = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            device_batch = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
                if torch.is_tensor(v)
            }
            if max_batches is not None and batch_index > max_batches:
                break
            if batch_index == 1 or batch_index % log_every == 0:
                _heartbeat(epoch_index + 1, start_epoch + epochs, batch_index, len(train_loader))

            optimizer.zero_grad(set_to_none=True)
            with _autocast(amp_enabled):
                loss_dict = _compute_losses(model, device_batch)
                loss = loss_dict["loss"]

            if not torch.isfinite(loss.detach()):
                components = []
                for name in ("mrl_loss", "sia_loss", "sia_attn_loss", "sia_cka_loss", "pic_loss"):
                    value = loss_dict.get(name)
                    if torch.is_tensor(value):
                        item = value.detach().float().item()
                        components.append(f"{name}={item}")
                detail = ", ".join(components)
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch_index + 1}, batch={batch_index}: "
                    f"{loss.detach().float().item()} ({detail})"
                )
            scaler.scale(loss.float()).backward()
            _clip_gradients(model, optimizer, scaler, amp_enabled, max_grad_norm)
            stepped = _optimizer_step(optimizer, scaler, amp_enabled, scheduler)

            batch_size = device_batch["input_ids1_stu"].size(0)
            total_loss += loss.detach().item() * batch_size
            total_mrl_loss += loss_dict["mrl_loss"].detach().item() * batch_size
            total_pic_loss += loss_dict["pic_loss"].detach().item() * batch_size
            item_count += batch_size

        total_loss, global_count = _distributed_average(total_loss, item_count, device)
        total_mrl_loss, _ = _distributed_average(total_mrl_loss, item_count, device)
        total_pic_loss, _ = _distributed_average(total_pic_loss, item_count, device)
        item_count = global_count

        history.append(
            {
                "total_loss": total_loss / max(1, item_count),
                "mrl_loss": total_mrl_loss / max(1, item_count),
                "pic_loss": total_pic_loss / max(1, item_count),
            }
        )

        if epoch_callback is not None:
            epoch_callback(epoch_index + 1, model, optimizer, scheduler, history)

    _disarm_stall_watchdog()
    return history




