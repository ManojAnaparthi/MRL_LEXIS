"""TweetEval Controlled Experiments Runner across MRL, MRL+SIA, MRL+PIC, and MIPIC.

Executes:
1. MRL Baseline (TinyBERT-6L + Matryoshka InfoNCE Loss, alpha=1.0)
2. MRL + SIA (TinyBERT-6L + MRL + SIA, alpha=0.4)
3. MRL + PIC (TinyBERT-6L + MRL + PIC, alpha=0.4, 5 PIC pairs)
4. MIPIC (TinyBERT-6L + MRL + SIA + PIC, alpha=0.4)

Each model is:
- Freshly initialized from pretrained TinyBERT-6L (no weight/state sharing)
- Trained for 5 epochs on the the same final-data corpus used by the controlled MRL baseline (global batch_size=16 (8/GPU on 2 GPUs), lr=2e-5, cosine schedule)
- Evaluated on TweetEval across all 7 MRL dimensions: [16, 32, 64, 128, 256, 512, 768]
- Checkpointed in results/checkpoints/<model_name>/
- Results saved to results/tweeteval_results.json and results/tweeteval_results.csv
"""

from __future__ import annotations

import argparse
import faulthandler
import functools
import json
import os
import signal
import sys
import time
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

print = functools.partial(print, flush=True)

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_scheduler

from evaluate import MRLEvaluationConfig, evaluate_tweeteval, validation_probe_score
from losses.mrl_loss import MRL_NESTED_DIMS
from models.mipic import MIPICConfig, TinyBERT_MIPIC, create_mipic_optimizer
from models.mrl_pic import TinyBERT_MRL_PIC, create_mrl_pic_optimizer
from models.mrl_sia import TinyBERT_MRL_SIA, create_mrl_sia_optimizer
from models.tinybert import MRL_READOUT, load_tinybert_6l, load_tinybert_tokenizer
from train import (
    MRLBaselineConfig,
    build_mrl_train_loader,
    train_mipic,
    train_mrl_baseline,
    train_mrl_pic,
    train_mrl_sia,
)

# Output paths
# Kaggle/local overrides are environment-driven so the existing CLI remains compatible.
# Example: MIPIC_OUTPUT_DIR=/kaggle/working/mipic_results
OUTPUT_ROOT = Path(os.environ.get("MIPIC_OUTPUT_DIR", "results"))
RESULTS_DIR = OUTPUT_ROOT
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
RESULTS_JSON = RESULTS_DIR / "tweeteval_results.json"
RESULTS_CSV = RESULTS_DIR / "tweeteval_results.csv"

def seed_everything(seed: int, rank: int = 0) -> int:
    """Make a run reproducible.

    Nothing was seeded before this, so two runs of an identical configuration
    produced different encoders: measured on 2xT4, the MRL arm scored 65.59 and
    62.52 macro-F1 at d=768 across two runs whose epoch-1 losses agreed to
    0.2003 vs 0.1977. Differences of that size were being read as signal.

    Each rank gets ``seed + rank`` so the two ranks' dropout masks decorrelate
    while the whole run stays reproducible. DistributedSampler already derives
    its permutation deterministically from its own seed plus the epoch.
    """
    import random as _random

    import numpy as _np

    effective = int(seed) + int(rank)
    _random.seed(effective)
    _np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    return effective


def _install_hang_diagnostics() -> None:
    """Make a stalled rank dump its own stack instead of hanging silently.

    A DDP deadlock otherwise shows up as "no CPU, no GPU, no output". With this
    installed, every rank prints a full traceback for every thread on SIGUSR1
    (`kill -USR1 <pid>`), and MIPIC_STACK_DUMP_SECONDS turns on a repeating
    automatic dump so a hang is self-diagnosing with no operator action.
    """
    faulthandler.enable(file=sys.stderr, all_threads=True)
    if hasattr(faulthandler, "register") and hasattr(signal, "SIGUSR1"):
        try:
            faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
        except (RuntimeError, ValueError):
            pass
    seconds = os.environ.get("MIPIC_STACK_DUMP_SECONDS", "").strip()
    if seconds and seconds not in {"0", ""}:
        try:
            faulthandler.dump_traceback_later(float(seconds), repeat=True, exit=False, file=sys.stderr)
        except (ValueError, RuntimeError):
            pass


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def distributed_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return distributed_rank() == 0


def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialize NCCL DDP when launched with torchrun on CUDA."""
    if "LOCAL_RANK" not in os.environ:
        if torch.cuda.is_available():
            return 0, 1, torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            return 0, 1, device
        torch.set_num_threads(4)
        return 0, 1, torch.device("cpu")

    if not torch.cuda.is_available():
        raise RuntimeError("torchrun DDP was requested but CUDA is unavailable.")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # torch.cuda.set_device() on an out-of-range ordinal raises a bare
    # "invalid device ordinal", which does not say why the device is missing.
    # The usual cause is a stale CUDA_VISIBLE_DEVICES restricting this process
    # to fewer GPUs than torchrun launched ranks for.
    visible = torch.cuda.device_count()
    if local_rank >= visible:
        raise RuntimeError(
            f"torchrun launched {world_size} ranks but this process can only see "
            f"{visible} CUDA device(s), so local_rank={local_rank} is out of range. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}. "
            "Unset it (or set it to '0,1') before launching 2-rank DDP; a "
            "single-GPU run should use `python run_experiments.py`, not torchrun."
        )

    torch.cuda.set_device(local_rank)
    # A finite timeout converts a stalled collective into a raised error with a
    # rank-attributed traceback, instead of an unbounded silent hang. NCCL's own
    # default is 10 minutes, but the watchdog only aborts when async error
    # handling is on, which _maybe_configure_kaggle() enables.
    timeout_minutes = float(os.environ.get("MIPIC_DDP_TIMEOUT_MIN", "20"))
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device("cuda", local_rank),
        timeout=timedelta(minutes=timeout_minutes),
    )
    if is_main_process():
        print(
            f"DDP: nccl, world_size={world_size}, collective timeout={timeout_minutes:g} min, "
            f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '<unset>')}, "
            f"NCCL_SHM_DISABLE={os.environ.get('NCCL_SHM_DISABLE', '<unset>')}",
            flush=True,
        )

    return rank, world_size, torch.device("cuda", local_rank)


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def maybe_wrap_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if not is_distributed():
        return model
    # MIPIC's loss graph contains 17 differentiable all_gather calls (7 MRL
    # prefixes + 5 PIC pairs x 2 views), whose backward passes issue their own
    # collectives interleaved with the DDP reducer's bucket all_reduces. If those
    # two streams are ever ordered differently on the two ranks, NCCL deadlocks:
    # output stops mid-epoch with no CPU or GPU activity. static_graph makes DDP
    # record the autograd execution order on the first iteration and replay it,
    # which pins that ordering. It is valid here because the graph and the
    # used/unused parameter set are identical on every step. Opt-in, because
    # changing DDP topology without evidence is not warranted; enable it with
    # MIPIC_DDP_STATIC_GRAPH=1 if a stack dump shows both ranks inside NCCL.
    static_graph = os.environ.get("MIPIC_DDP_STATIC_GRAPH", "0").lower() in {"1", "true", "yes"}
    if static_graph and is_main_process():
        print("DDP: static_graph=True (deterministic reducer ordering)", flush=True)
    return DDP(
        model,
        device_ids=[device.index],
        output_device=device.index,
        static_graph=static_graph,
        # Every parameter receives a gradient on every step: the encoder, the
        # pooler (it is z, consumed by L_MRL), the SIA projections P_i and all
        # five PIC projectors. The unused indices 101/102 reported historically
        # were the pooler's two tensors, back when L_MRL read the raw [CLS];
        # see tests/test_ddp_two_rank.py.
        find_unused_parameters=False,
        broadcast_buffers=False,
    )


# Dataset paths
def _resolve_data_file(filename: str, env_var: str) -> Path:
    """Locate a bundled data file: explicit env var, then MIPIC_DATA_DIR, then
    the repository's data/, then a search under /kaggle/input."""
    configured = os.environ.get(env_var, "").strip()
    if configured:
        return Path(configured)
    data_dir = os.environ.get("MIPIC_DATA_DIR", "").strip()
    if data_dir and (Path(data_dir) / filename).exists():
        return Path(data_dir) / filename
    local = Path("data") / filename
    if local.exists():
        return local
    if Path("/kaggle/input").exists():
        matches = sorted(Path("/kaggle/input").rglob(filename))
        if matches:
            return matches[0]
    return local


# THE configuration. One corpus, one probe, no presets and no switches.
#
# Corpus: paper Appendix A.2 -- sentences pooled from every source, used whole.
# Probe : the 3,000 sampled TweetEval rows (Table 6) and the full test split.
TRAIN_CORPUS_CSV = _resolve_data_file("final_data.csv", "MIPIC_TRAIN_CORPUS")
# The full TweetEval train split (26,732 rows). tweet_train_3k.csv is the
# 3,000-row slice that went into the *training corpus*, not a probe split.
EVAL_TRAIN_CSV = _resolve_data_file("tweet_train.csv", "MIPIC_EVAL_TRAIN")
EVAL_TEST_CSV = _resolve_data_file("tweet_test.csv", "MIPIC_EVAL_TEST")


def save_results(results: Dict[str, object]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    rows: List[Dict[str, object]] = []
    if "evaluations" in results:
        for model_name, dim_dict in results["evaluations"].items():
            for dim, metrics in dim_dict.items():
                rows.append({
                    "Model": model_name,
                    "Dimension": int(dim),
                    "Accuracy": metrics["accuracy"],
                    "Macro_F1": metrics["f1"],
                })
    if rows:
        df = pd.DataFrame(rows).sort_values(by=["Model", "Dimension"]).reset_index(drop=True)
        df.to_csv(RESULTS_CSV, index=False)


def load_existing_results() -> Dict[str, object]:
    if RESULTS_JSON.exists():
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"metadata": {}, "training_history": {}, "evaluations": {}}


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    num_batches: int,
    epochs: int,
    warmup_ratio: float = 0.1,
    min_lr: float = 2e-6,
):
    """Build the cosine scheduler (Table 7) used identically by all four arms.

    Table 7 specifies "Cosine" without a warmup fraction. The official notebook
    uses ``warmup_ratio = 0.1``, and the repository's own MRL baseline path
    (``create_mrl_optimizer_scheduler``) already used 0.1 while the SIA/PIC/MIPIC
    arms used 0. That asymmetry made the controlled comparison unfair, so all
    four arms now share this builder and this warmup fraction.
    """
    total_steps = num_batches * epochs
    return get_scheduler(
        # cosine_with_min_lr, not plain cosine: the official notebook and the
        # repository's best-scoring run both floor the schedule at 2e-6 rather
        # than annealing to zero.
        name="cosine_with_min_lr",
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
        scheduler_specific_kwargs={"min_lr": min_lr},
    )


def print_comparison_table(results: Dict[str, object]) -> None:
    evals = results.get("evaluations", {})
    if not evals:
        return

    metadata = results.get("metadata", {}) or {}
    if metadata.get("truncated_run"):
        seen = metadata.get("train_sentences_seen", "?")
        print("\n" + "!" * 90)
        print(f"!! PLUMBING CHECK ONLY - the encoder saw {seen} sentences "
              f"(MIPIC_MAX_BATCHES={metadata.get('max_batches_per_rank')}).")
        print("!! The table below is NOT a result. Unset MIPIC_MAX_BATCHES for a real run.")
        print("!" * 90)

    models = [m for m in ["mrl", "mrl_sia", "mrl_pic", "mipic"] if m in evals]
    print("\n" + "=" * 90)
    print("TWEETEVAL EVALUATION RESULTS ACROSS ALL 7 MRL DIMENSIONS")
    print("=" * 90)

    header = f"{'Dimension':^10} | " + " | ".join(f"{m.upper() + ' (Acc / F1)':^18}" for m in models)
    print(header)
    print("-" * len(header))

    for dim in MRL_NESTED_DIMS:
        row = [f"{dim:^10}"]
        for m in models:
            metrics = evals[m].get(str(dim)) or evals[m].get(dim)
            if metrics:
                acc = metrics["accuracy"] * 100
                f1 = metrics["f1"] * 100
                row.append(f"{acc:6.2f}% / {f1:6.2f}%")
            else:
                row.append("      N/A        ")
        print(" | ".join(row))
    print("=" * 90 + "\n")



def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _save_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    epoch: int,
    history: List[Dict[str, float]],
    device: torch.device,
) -> None:
    """Save everything needed to resume after a Kaggle session interruption."""
    state = {
        "epoch": epoch,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "history": history,
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _load_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    device: torch.device,
) -> tuple[int, List[Dict[str, float]]]:
    """Restore a resumable checkpoint and return the next epoch plus history."""
    state = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if state.get("scheduler_state_dict") is not None and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if state.get("rng_state") is not None:
        torch.random.set_rng_state(state["rng_state"])
    if device.type == "cuda" and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return int(state["epoch"]), list(state.get("history", []))


def _resume_checkpoint_path(model_name: str) -> Path:
    return (CHECKPOINT_DIR / model_name / "training_state.pt").resolve()


def _completed_checkpoint_path(model_name: str) -> Path:
    return (CHECKPOINT_DIR / model_name / "model.pt").resolve()


def _configured_checkpoint_path(model_name: str, eval_only: bool) -> Path:
    configured = os.environ.get("MIPIC_CHECKPOINT")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"MIPIC_CHECKPOINT does not exist: {path}")

    completed = _completed_checkpoint_path(model_name)
    if completed.exists():
        return completed

    resume = _resume_checkpoint_path(model_name)
    if eval_only and resume.exists():
        return resume

    # Accept the historical relative path as a last-resort compatibility
    # location, but never silently change the configured output directory.
    legacy = Path("results") / "checkpoints" / model_name / "model.pt"
    if legacy.exists():
        return legacy.resolve()

    raise FileNotFoundError(
        f"No checkpoint found for {model_name!r}. Checked {completed}, "
        f"{resume}, and {legacy}. Set MIPIC_CHECKPOINT explicitly if needed."
    )


def _atomic_torch_save(obj: object, path: Path) -> None:
    """Atomically publish a torch checkpoint so partial files are never exposed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(obj, temp_path)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _assert_checkpoint(path: Path) -> None:
    if not path.exists():
        raise IOError(f"Checkpoint was not created: {path}")
    size = path.stat().st_size
    if size <= 1024:
        raise IOError(f"Checkpoint is unexpectedly small ({size} bytes): {path}")

def _load_model_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    """Load either a completed model.pt or a resumable training_state.pt."""
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    model.load_state_dict(state)


def _maybe_configure_kaggle() -> None:
    """Set conservative Kaggle/NCCL defaults without changing local behavior.

    These must be applied before ``init_process_group``; NCCL reads its
    environment once, at communicator creation.
    """
    # A forked DataLoader worker that inherits an already-parallel Rust
    # tokenizer is a documented deadlock source. Safe everywhere, so it is not
    # gated on Kaggle.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Abort a stuck collective via the NCCL watchdog instead of blocking forever.
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        # Keep Hugging Face downloads/cache under the writable Kaggle working tree.
        os.environ.setdefault("HF_HOME", "/kaggle/working/huggingface")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/kaggle/working/huggingface/transformers")
        # Kaggle's paired T4s have no NVLink and their PCIe peer-to-peer path is
        # a known source of intermittent NCCL stalls. Disabling P2P costs
        # essentially nothing for a 67M-parameter model over two ranks.
        # Escalation if a hang persists: NCCL_SHM_DISABLE=1 (Kaggle's /dev/shm
        # is small and is shared with DataLoader workers).
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")


def _validate_evaluation_inputs() -> None:
    """Fail fast before training if evaluation CSVs do not have text/label columns."""
    required = {"text", "label"}
    problems = []
    for path in (EVAL_TRAIN_CSV, EVAL_TEST_CSV):
        try:
            frame = pd.read_csv(path, nrows=1)
        except Exception as exc:
            problems.append(f"{path}: cannot read CSV ({exc})")
            continue
        missing = required.difference(frame.columns)
        if missing:
            problems.append(f"{path}: missing columns {sorted(missing)}")
    if problems:
        raise ValueError("Invalid evaluation inputs; refusing to start training:\n" + "\n".join(f"  - {p}" for p in problems))


def _best_epoch_path(model_name: str) -> Path:
    return (CHECKPOINT_DIR / model_name / "best.pt").resolve()


def _make_selector(model_name, encoder_of, tokenizer, eval_config):
    """Score the encoder on a held-out slice of the probe-train split each epoch.

    The contrastive objective is solved inside one epoch (MRL loss 0.200 ->
    0.0049 by epoch 2), so epochs 3-5 apply gradient to a solved task and
    degrade the encoder: measured 62.52 macro-F1 at d=768 against 70.55 for the
    untrained backbone. This keeps the best epoch instead of the last one.

    Selection uses ONLY probe-train rows, never the test set.
    """
    state = {"best_score": float("-inf"), "best_epoch": 0, "history": []}
    best_path = _best_epoch_path(model_name)

    def on_epoch(epoch, model):
        score = validation_probe_score(encoder_of(model), tokenizer, EVAL_TRAIN_CSV, eval_config)
        state["history"].append({"epoch": epoch, "val_macro_f1": round(score * 100, 3)})
        marker = ""
        if score > state["best_score"]:
            state["best_score"], state["best_epoch"] = score, epoch
            _atomic_torch_save(unwrap_model(model).state_dict(), best_path)
            marker = "  <- best so far, checkpointed"
        print(f"Epoch {epoch}: held-out probe macro-F1 {score * 100:.2f}{marker}", flush=True)

    return on_epoch, state


def run_experiment(
    model_to_run: Optional[str] = None,
    eval_only: bool = False,
    skip_existing: bool = False,
) -> None:
    _maybe_configure_kaggle()
    _install_hang_diagnostics()

    # Resolve bundled evaluation files before initializing the expensive training
    # path. This keeps Kaggle execution independent of the notebook cwd.
    global EVAL_TRAIN_CSV, EVAL_TEST_CSV
    if not EVAL_TRAIN_CSV.exists() and Path("/kaggle/input").exists():
        matches = sorted(Path("/kaggle/input").rglob(EVAL_TRAIN_CSV.name))
        if matches:
            EVAL_TRAIN_CSV = matches[0]
    if not EVAL_TEST_CSV.exists() and Path("/kaggle/input").exists():
        matches = sorted(Path("/kaggle/input").rglob(EVAL_TEST_CSV.name))
        if matches:
            EVAL_TEST_CSV = matches[0]

    rank, world_size, device = setup_distributed()

    seed = _env_int("MIPIC_SEED", 42)
    effective_seed = seed_everything(seed, rank)
    if is_main_process():
        print(f"Seed: {seed} (rank-adjusted per process; this rank uses {effective_seed})", flush=True)

    if is_main_process():
        if device.type == "cuda":
            print(
                f"Device: {device} | distributed_world_size={world_size} | "
                f"GPU: {torch.cuda.get_device_name(device.index)}"
            )
        elif device.type == "mps":
            print("Device: mps (Apple Silicon GPU)")
        else:
            print(f"Device: {device}, PyTorch CPU threads: {torch.get_num_threads()}")

    assert TRAIN_CORPUS_CSV.exists(), f"Missing {TRAIN_CORPUS_CSV}"
    assert EVAL_TRAIN_CSV.exists(), f"Missing {EVAL_TRAIN_CSV}"
    assert EVAL_TEST_CSV.exists(), f"Missing {EVAL_TEST_CSV}"
    # This is a tiny header-only check and intentionally runs on every rank so
    # a bad evaluation artifact fails before any expensive forward/backward pass.
    _validate_evaluation_inputs()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    config = MRLBaselineConfig(
        batch_size=_env_int(
            "MIPIC_BATCH_SIZE",
            8 if world_size > 1 else 16,
        ),
        epochs=_env_int("MIPIC_EPOCHS", 5),
        learning_rate=float(os.environ.get("MIPIC_LEARNING_RATE", "2e-5")),
        max_length=_env_int("MIPIC_MAX_LENGTH", 256),
        temperature=0.05,
        nested_dims=MRL_NESTED_DIMS,
        warmup_ratio=0.1,
        min_learning_rate=2e-6,
        # DataLoader worker processes share Kaggle's small /dev/shm with the
        # NCCL transport, and the collate function holds a Rust tokenizer. Both
        # are documented hang sources under torchrun, and tokenising short
        # sentences is cheap next to SIA's 36 CKA terms, so the distributed
        # default is in-process loading.
        num_workers=_env_int("MIPIC_NUM_WORKERS", 0),
        max_grad_norm=0.0,
    )
    eval_config = MRLEvaluationConfig(
        dimensions=MRL_NESTED_DIMS,
        batch_size=_env_int("MIPIC_EVAL_BATCH_SIZE", 64),
        max_length=config.max_length,
        # Whole split by default; the env var exists only to make the
        # sub-minute pipeline check cheap.
        train_samples=(
            int(os.environ["MIPIC_EVAL_TRAIN_SAMPLES"])
            if os.environ.get("MIPIC_EVAL_TRAIN_SAMPLES")
            else None
        ),
        random_state=_env_int("MIPIC_RANDOM_STATE", 42),
        max_iter=1000,
        test_samples=(
            int(os.environ["MIPIC_EVAL_TEST_SAMPLES"])
            if os.environ.get("MIPIC_EVAL_TEST_SAMPLES")
            else None
        ),
    )

    all_results = load_existing_results()
    all_results["metadata"] = {
        "device": str(device),
        "epochs": config.epochs,
        "batch_size_per_gpu": config.batch_size,
        "global_batch_size": config.batch_size * world_size,
        "learning_rate": config.learning_rate,
        "temperature": config.temperature,
        "sia_temperature": MIPICConfig().sia_temperature,
        "mrl_readout": MRL_READOUT,
        "pic_detach_target": MIPICConfig().pic_detach_target,
        "max_grad_norm": config.max_grad_norm,
        "pic_temperature": MIPICConfig().pic_temperature,
        "alpha": MIPICConfig().alpha,
        "max_grad_norm": config.max_grad_norm,
        "warmup_ratio": config.warmup_ratio,
        "scheduler": "cosine",
        "probe_max_iter": eval_config.max_iter,
        "dimensions": list(MRL_NESTED_DIMS),
        "train_corpus": str(TRAIN_CORPUS_CSV),
        "train_samples": len(pd.read_csv(TRAIN_CORPUS_CSV)),
        # A single-domain corpus (e.g. tweet_train_3k.csv) has no 'source'
        # column; record that rather than crashing, so corpus choice is always
        # visible in the results file.
        "train_sources": (
            pd.read_csv(TRAIN_CORPUS_CSV)["source"].value_counts().to_dict()
            if "source" in pd.read_csv(TRAIN_CORPUS_CSV, nrows=1).columns
            else {"<single-domain corpus, no source column>": len(pd.read_csv(TRAIN_CORPUS_CSV))}
        ),
        "eval_train_samples": len(pd.read_csv(EVAL_TRAIN_CSV)),
        "eval_test_samples": len(pd.read_csv(EVAL_TEST_CSV)),
        "output_dir": str(RESULTS_DIR),
        "kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")),
        "seed": seed,
        "eval_train_csv": str(EVAL_TRAIN_CSV),
        "eval_test_csv": str(EVAL_TEST_CSV),
    }
    # Filled in below once truncation is known; kept in the results file so a
    # smoke run can never be mistaken for a reportable one after the fact.
    all_results["metadata"]["truncated_run"] = False

    target_models = [model_to_run] if model_to_run else ["mrl", "mrl_sia", "mrl_pic", "mipic"]
    fresh = os.environ.get("MIPIC_FRESH", "0").lower() in {"1", "true", "yes"}
    # MIPIC_MAX_BATCHES truncates training. By default such a run is treated as
    # a pure forward/backward smoke test and neither checkpoints nor evaluates.
    # MIPIC_SMOKE_FULL_PIPELINE=1 keeps the truncation but still exercises
    # train -> save -> assert non-empty -> evaluate, so the checkpoint and
    # evaluation paths can be validated in minutes instead of only after a full
    # GPU run.
    full_pipeline = os.environ.get("MIPIC_SMOKE_FULL_PIPELINE", "0").lower() in {"1", "true", "yes"}
    max_batches = os.environ.get("MIPIC_MAX_BATCHES", "").strip()
    truncated = max_batches not in {"", "0"}
    smoke_mode = truncated and not full_pipeline

    if truncated:
        all_results["metadata"]["truncated_run"] = True
        all_results["metadata"]["max_batches_per_rank"] = int(max_batches)
        all_results["metadata"]["train_sentences_seen"] = (
            int(max_batches) * config.batch_size * world_size * config.epochs
        )
    if truncated and is_main_process():
        per_rank = int(max_batches) * config.batch_size
        seen = per_rank * world_size
        corpus_rows = all_results["metadata"]["train_samples"]
        print("\n" + "!" * 78)
        print("!! TRUNCATED RUN - MIPIC_MAX_BATCHES=" + max_batches)
        print(f"!! Trains on {seen} sentences ({max_batches} batches x "
              f"{config.batch_size}/rank x {world_size} rank(s)), "
              f"{seen / max(1, corpus_rows) / config.epochs * 100:.2f}% of the "
              f"{corpus_rows:,}-row corpus per epoch.")
        if full_pipeline:
            print("!! MIPIC_SMOKE_FULL_PIPELINE=1: a checkpoint and an evaluation WILL be")
            print("!! produced. They validate plumbing only. The metrics are NOT results.")
        print("!! Unset MIPIC_MAX_BATCHES for a real run.")
        print("!" * 78 + "\n", flush=True)

    for model_name in target_models:
        ckpt_dir = CHECKPOINT_DIR / model_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = _completed_checkpoint_path(model_name)
        resume_path = _resume_checkpoint_path(model_name)
        if eval_only:
            ckpt_path = _configured_checkpoint_path(model_name, eval_only=True)

        if skip_existing and ckpt_path.exists() and model_name in all_results.get("evaluations", {}):
            if is_main_process():
                print(f"Skipping {model_name.upper()}: already completed and evaluated.")
            if is_distributed():
                dist.barrier()
            continue
        if fresh and is_main_process():
            for stale_path in (resume_path, ckpt_path):
                if stale_path.exists():
                    stale_path.unlink()
            all_results.get("training_history", {}).pop(model_name, None)
            all_results.get("evaluations", {}).pop(model_name, None)
        if is_distributed():
            dist.barrier()

        print(f"\n{'=' * 70}")
        mode_str = "EVALUATION ONLY" if eval_only else "TRAINING & EVALUATION"
        print(f"STARTING EXPERIMENT: {model_name.upper()} ({mode_str})")
        print(f"{'=' * 70}")

        # A tokenizer is needed for both training and evaluation.  In eval-only
        # mode, the checkpoint is loaded into a freshly constructed model and
        # no training dataloader or optimizer is created.
        tokenizer = load_tinybert_tokenizer()
        t_start = time.time()
        selector = None
        _trained_model = None

        if model_name == "mrl":
            if eval_only:
                raw_model, _ = load_tinybert_6l(device=device)
                _load_model_checkpoint(raw_model, ckpt_path, device)
                trained_encoder = raw_model
                _trained_model = raw_model
                history = all_results.get("training_history", {}).get(model_name, [])
            else:
                train_loader = build_mrl_train_loader(TRAIN_CORPUS_CSV, tokenizer, config)
                raw_model, _ = load_tinybert_6l(device=device)
                start_epoch = 0
                history_prefix: List[Dict[str, float]] = []
                # Same optimizer/scheduler construction as the other three arms.
                optimizer = torch.optim.AdamW(
                    [p for p in raw_model.parameters() if p.requires_grad],
                    lr=config.learning_rate,
                    weight_decay=0.01,
                )
                scheduler = build_cosine_scheduler(
                    optimizer, len(train_loader), config.epochs, config.warmup_ratio
                )
                if resume_path.exists():
                    start_epoch, history_prefix = _load_training_state(resume_path, raw_model, optimizer, scheduler, device)
                    print(f"Resuming {model_name.upper()} from epoch {start_epoch}/{config.epochs}.")
                raw_model = maybe_wrap_ddp(raw_model, device)
                _select_epoch, selector = _make_selector(model_name, lambda m: unwrap_model(m), tokenizer, eval_config)
                def _checkpoint(epoch, model, opt, sch, current_history):
                    _save_training_state(resume_path, model, opt, sch, epoch, history_prefix + current_history, device)
                    print(f"Checkpoint: epoch {epoch}/{config.epochs}")
                    _select_epoch(epoch, model)
                new_history = train_mrl_baseline(
                    raw_model,
                    train_loader,
                    config,
                    device,
                    start_epoch=start_epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch_callback=None if (smoke_mode or not is_main_process()) else _checkpoint,
                )
                history = history_prefix + new_history
                trained_encoder = unwrap_model(raw_model)
                _trained_model = trained_encoder
                if not smoke_mode and is_main_process():
                    _atomic_torch_save(unwrap_model(raw_model).state_dict(), ckpt_path)
                    _assert_checkpoint(ckpt_path)

        elif model_name == "mrl_sia":
            mrl_sia = TinyBERT_MRL_SIA(device=device)
            if eval_only:
                _load_model_checkpoint(mrl_sia, ckpt_path, device)
                history = all_results.get("training_history", {}).get(model_name, [])
            else:
                train_loader = build_mrl_train_loader(TRAIN_CORPUS_CSV, tokenizer, config)
                optimizer = create_mrl_sia_optimizer(mrl_sia)
                scheduler = build_cosine_scheduler(optimizer, len(train_loader), config.epochs)
                start_epoch = 0
                history_prefix: List[Dict[str, float]] = []
                if resume_path.exists():
                    start_epoch, history_prefix = _load_training_state(resume_path, mrl_sia, optimizer, scheduler, device)
                    if is_main_process():
                        print(f"Resuming {model_name.upper()} from epoch {start_epoch}/{config.epochs}.")
                mrl_sia = maybe_wrap_ddp(mrl_sia, device)
                _select_epoch, selector = _make_selector(model_name, lambda m: unwrap_model(m).encoder, tokenizer, eval_config)
                def _checkpoint(epoch, model, opt, sch, current_history):
                    _save_training_state(resume_path, model, opt, sch, epoch, history_prefix + current_history, device)
                    print(f"Checkpoint: epoch {epoch}/{config.epochs}")
                    _select_epoch(epoch, model)
                new_history = train_mrl_sia(
                    mrl_sia,
                    train_loader,
                    device,
                    epochs=max(0, config.epochs - start_epoch),
                    start_epoch=start_epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch_callback=None if (smoke_mode or not is_main_process()) else _checkpoint,
                )
                history = history_prefix + new_history
                if not smoke_mode and is_main_process():
                    _atomic_torch_save(unwrap_model(mrl_sia).state_dict(), ckpt_path)
                    _assert_checkpoint(ckpt_path)
            _trained_model = unwrap_model(mrl_sia)
            trained_encoder = _trained_model.encoder

        elif model_name == "mrl_pic":
            mrl_pic = TinyBERT_MRL_PIC(device=device)
            if eval_only:
                _load_model_checkpoint(mrl_pic, ckpt_path, device)
                history = all_results.get("training_history", {}).get(model_name, [])
            else:
                train_loader = build_mrl_train_loader(TRAIN_CORPUS_CSV, tokenizer, config)
                optimizer = create_mrl_pic_optimizer(mrl_pic)
                scheduler = build_cosine_scheduler(optimizer, len(train_loader), config.epochs)
                start_epoch = 0
                history_prefix: List[Dict[str, float]] = []
                if resume_path.exists():
                    start_epoch, history_prefix = _load_training_state(resume_path, mrl_pic, optimizer, scheduler, device)
                    if is_main_process():
                        print(f"Resuming {model_name.upper()} from epoch {start_epoch}/{config.epochs}.")
                mrl_pic = maybe_wrap_ddp(mrl_pic, device)
                _select_epoch, selector = _make_selector(model_name, lambda m: unwrap_model(m).encoder, tokenizer, eval_config)
                def _checkpoint(epoch, model, opt, sch, current_history):
                    _save_training_state(resume_path, model, opt, sch, epoch, history_prefix + current_history, device)
                    print(f"Checkpoint: epoch {epoch}/{config.epochs}")
                    _select_epoch(epoch, model)
                new_history = train_mrl_pic(
                    mrl_pic,
                    train_loader,
                    device,
                    epochs=max(0, config.epochs - start_epoch),
                    start_epoch=start_epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch_callback=None if (smoke_mode or not is_main_process()) else _checkpoint,
                )
                history = history_prefix + new_history
                if not smoke_mode and is_main_process():
                    _atomic_torch_save(unwrap_model(mrl_pic).state_dict(), ckpt_path)
                    _assert_checkpoint(ckpt_path)
            _trained_model = unwrap_model(mrl_pic)
            trained_encoder = _trained_model.encoder

        elif model_name == "mipic":
            mipic = TinyBERT_MIPIC(device=device)
            if eval_only:
                _load_model_checkpoint(mipic, ckpt_path, device)
                history = all_results.get("training_history", {}).get(model_name, [])
            else:
                train_loader = build_mrl_train_loader(TRAIN_CORPUS_CSV, tokenizer, config)
                optimizer = create_mipic_optimizer(mipic)
                scheduler = build_cosine_scheduler(optimizer, len(train_loader), config.epochs)
                start_epoch = 0
                history_prefix: List[Dict[str, float]] = []
                if resume_path.exists():
                    start_epoch, history_prefix = _load_training_state(resume_path, mipic, optimizer, scheduler, device)
                    if is_main_process():
                        print(f"Resuming {model_name.upper()} from epoch {start_epoch}/{config.epochs}.")
                mipic = maybe_wrap_ddp(mipic, device)
                _select_epoch, selector = _make_selector(model_name, lambda m: unwrap_model(m).encoder, tokenizer, eval_config)
                def _checkpoint(epoch, model, opt, sch, current_history):
                    _save_training_state(resume_path, model, opt, sch, epoch, history_prefix + current_history, device)
                    print(f"Checkpoint: epoch {epoch}/{config.epochs}")
                    _select_epoch(epoch, model)
                new_history = train_mipic(
                    mipic,
                    train_loader,
                    device,
                    epochs=max(0, config.epochs - start_epoch),
                    start_epoch=start_epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch_callback=None if (smoke_mode or not is_main_process()) else _checkpoint,
                )
                history = history_prefix + new_history
                if not smoke_mode and is_main_process():
                    _atomic_torch_save(unwrap_model(mipic).state_dict(), ckpt_path)
                    _assert_checkpoint(ckpt_path)
            _trained_model = unwrap_model(mipic)
            trained_encoder = _trained_model.encoder

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        if smoke_mode and not eval_only:
            print("\nSmoke test completed: partial training only; no completion checkpoint or evaluation was recorded.", flush=True)
            continue

        if is_distributed():
            dist.barrier()

        if not eval_only and is_main_process():
            t_train = time.time() - t_start
            print(f"\nTraining completed in {t_train:.1f}s ({t_train / 60:.2f} min)")
            print(f"Checkpoint saved to {ckpt_path}")
            for epoch_idx, epoch_stats in enumerate(history, 1):
                stats_str = ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
                print(f"  Epoch {epoch_idx}/{config.epochs}: {stats_str}")
            all_results["training_history"][model_name] = history

        # -------------------------------------------------------------
        # Evaluate on TweetEval linear probe across all 7 dimensions.
        # Rank 0 does the expensive evaluation; all ranks participate in
        # the error broadcast so a rank-0 exception cannot strand peers
        # in a NCCL barrier until the watchdog timeout.
        # -------------------------------------------------------------
        eval_error = None
        eval_metrics = None
        final_metrics = None
        best_path = _best_epoch_path(model_name)
        selected_epoch = selector["best_epoch"] if (not eval_only and selector) else None
        t_eval_start = time.time()
        if is_main_process():
            print(f"\nRunning TweetEval Linear Probe evaluation for {model_name}...")
            try:
                # The last epoch, i.e. the paper-faithful 5-epoch model.
                final_metrics = evaluate_tweeteval(
                    model=trained_encoder, tokenizer=tokenizer,
                    train_csv=EVAL_TRAIN_CSV, test_csv=EVAL_TEST_CSV, config=eval_config,
                )
                eval_metrics = final_metrics
                # The epoch the held-out probe preferred. Selection used only
                # probe-train rows, so this does not leak the test set.
                if (selected_epoch and selected_epoch != config.epochs
                        and best_path.exists() and _trained_model is not None):
                    print(f"Held-out probe preferred epoch {selected_epoch}/{config.epochs}; "
                          f"evaluating that checkpoint too.", flush=True)
                    _load_model_checkpoint(_trained_model, best_path, device)
                    eval_metrics = evaluate_tweeteval(
                        model=trained_encoder, tokenizer=tokenizer,
                        train_csv=EVAL_TRAIN_CSV, test_csv=EVAL_TEST_CSV, config=eval_config,
                    )
            except Exception as exc:
                eval_error = f"{type(exc).__name__}: {exc}"

        if is_distributed():
            error_box = [eval_error]
            dist.broadcast_object_list(error_box, src=0)
            eval_error = error_box[0]

        if eval_error is not None:
            # Never turn a post-training evaluation failure into a lost training
            # run. The completed checkpoint and resumable checkpoint remain on
            # disk, and the error is recorded for diagnosis.
            if is_main_process():
                t_eval = time.time() - t_eval_start
                print(f"Evaluation FAILED after successful training in {t_eval:.1f}s: {eval_error}", flush=True)
                all_results.setdefault("evaluation_errors", {})[model_name] = eval_error
                save_results(all_results)
            continue

        if is_main_process():
            t_eval = time.time() - t_eval_start
            print(f"Evaluation completed in {t_eval:.1f}s ({t_eval / 60:.2f} min)")
            all_results["evaluations"][model_name] = {
                str(dim): metrics for dim, metrics in eval_metrics.items()
            }
            if final_metrics is not None and eval_metrics is not final_metrics:
                all_results["evaluations"][f"{model_name}_last_epoch"] = {
                    str(dim): metrics for dim, metrics in final_metrics.items()
                }
            if selector:
                all_results.setdefault("epoch_selection", {})[model_name] = {
                    "selected_epoch": selector["best_epoch"],
                    "total_epochs": config.epochs,
                    "validation_curve": selector["history"],
                }
            save_results(all_results)
            # Only delete the resumable checkpoint after both model.pt and the
            # evaluation results have been successfully persisted.
            if not eval_only and resume_path.exists():
                resume_path.unlink()
            print_comparison_table(all_results)

        if is_distributed():
            dist.barrier()

    if is_main_process():
        print("\n" + "=" * 70)
        print("ALL REQUESTED EXPERIMENTS COMPLETED!")
        print(f"Results saved to: {RESULTS_JSON} and {RESULTS_CSV}")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TweetEval experiments.")
    parser.add_argument("--model", type=str, choices=["mrl", "mrl_sia", "mrl_pic", "mipic", "all"], default="all")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing checkpoint without training.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip models that are already trained and evaluated.")
    args = parser.parse_args()

    selected = None if args.model == "all" else args.model
    try:
        run_experiment(selected, eval_only=args.eval_only, skip_existing=args.skip_existing)
    finally:
        cleanup_distributed()
