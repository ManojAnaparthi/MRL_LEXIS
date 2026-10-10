#!/usr/bin/env python3

import os
import gc
import json
import math
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_scheduler,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "LingoIITGN/FlowVec-v1"

DIMENSIONS = [
    16,
    32,
    64,
    128,
    256,
    512,
    768,
    1024,
    2048,
    4096,
]

MAX_LENGTH = 256

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1

EPOCHS = 5
LR = 2e-5
MIN_LR = 2e-6
WARMUP_RATIO = 0.10

TEMPERATURE = 0.05

SEED = 42

# Number of dataloader workers
NUM_WORKERS = 2

# Save every epoch
SAVE_EVERY_EPOCH = True


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data",
    type=str,
    default="data/final_data.csv",
)

parser.add_argument(
    "--output",
    type=str,
    default="results/flowvec_mrl",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=EPOCHS,
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=BATCH_SIZE,
)

parser.add_argument(
    "--grad-accum",
    type=int,
    default=GRAD_ACCUM_STEPS,
)

parser.add_argument(
    "--max-length",
    type=int,
    default=MAX_LENGTH,
)

parser.add_argument(
    "--lr",
    type=float,
    default=LR,
)

parser.add_argument(
    "--smoke-test",
    action="store_true",
)

args = parser.parse_args()


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "Run this script inside the IITGN GPU allocation."
    )

print("=" * 70)
print("FLOWVEC A2 MRL TRAINING")
print("=" * 70)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "GB",
)

device = torch.device("cuda")


# ============================================================
# DATASET
# ============================================================

class TextPairDataset(Dataset):

    def __init__(self, dataframe):
        self.texts = dataframe["text"].astype(str).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]

        # Same-sentence positive pair exactly as A1.
        return {
            "sentence1": text,
            "sentence2": text,
        }


# ============================================================
# COLLATOR
# ============================================================

class FlowVecCollator:

    def __init__(self, tokenizer, max_length):

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):

        sentences1 = [
            item["sentence1"]
            for item in batch
        ]

        sentences2 = [
            item["sentence2"]
            for item in batch
        ]

        enc1 = self.tokenizer(
            sentences1,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        enc2 = self.tokenizer(
            sentences2,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        return {
            "input_ids1": enc1["input_ids"],
            "attention_mask1": enc1["attention_mask"],

            "input_ids2": enc2["input_ids"],
            "attention_mask2": enc2["attention_mask"],
        }


# ============================================================
# FLOWVEC POOLING
# ============================================================

def last_token_pool(
    hidden_states,
    attention_mask,
):
    """
    FlowVec/Qwen-style last-token pooling.

    hidden_states:
        [B, L, 4096]

    attention_mask:
        [B, L]

    returns:
        [B, 4096]
    """

    last_token_indices = (
        attention_mask.sum(dim=1) - 1
    )

    batch_indices = torch.arange(
        hidden_states.size(0),
        device=hidden_states.device,
    )

    embeddings = hidden_states[
        batch_indices,
        last_token_indices,
    ]

    return embeddings


# ============================================================
# MATRYOSHKA INFONCE
# ============================================================

def Matry_infonce(
    a,
    b,
    temperature=0.05,
    nested_dims=None,
):

    if nested_dims is None:
        nested_dims = DIMENSIONS

    assert a.dim() == 2
    assert b.dim() == 2
    assert a.shape == b.shape

    total_loss = 0.0

    for dim in nested_dims:

        if dim > a.size(-1):
            continue

        # Prefix representation
        q = a[:, :dim]
        k = b[:, :dim]

        # Same normalization as A1
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # In-batch similarity
        logits = torch.matmul(q, k.T)

        logits = logits / temperature
        if q.size(0) == 2:
            print(
                f"\nDEBUG dim={dim}"
                f" q_norm={q.norm(dim=1)}"
                f" k_norm={k.norm(dim=1)}"
                f" logits=\n{logits.float().detach().cpu()}"
            )
        # Positive pair is diagonal
        labels = torch.arange(
            q.size(0),
            device=q.device,
        )

        loss = F.cross_entropy(
            logits,
            labels,
        )

        if q.size(0) == 2:
            print(
                f"DEBUG dim={dim}"
                f" | loss={loss.item():.10f}"
                f" | labels={labels.detach().cpu().tolist()}"
            )


        total_loss = total_loss + loss

    return total_loss


# ============================================================
# MODEL LOADING
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    subfolder="0_Transformer",
)

print("Tokenizer loaded.")


print("\nLoading FlowVec-v1...")

# BF16 is preferable on L40S.
model = AutoModel.from_pretrained(
    MODEL_ID,
    subfolder="0_Transformer",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

model.to(device)

print("Model loaded.")

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters()) / 1e9,
    "B",
)


# ============================================================
# MEMORY OPTIMIZATION
# ============================================================

# Gradient checkpointing dramatically reduces activation memory.
if hasattr(model, "gradient_checkpointing_enable"):

    print("Enabling gradient checkpointing...")

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )

    print("Gradient checkpointing enabled.")


# Disable cache if present.
if hasattr(model, "config"):

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False


# ============================================================
# TRAINABLE PARAMETERS
# ============================================================

for param in model.parameters():
    param.requires_grad = True

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("\nTrainable parameters:")
print(
    f"{trainable_params / 1e9:.3f}B / "
    f"{total_params / 1e9:.3f}B"
)


# ============================================================
# DATA
# ============================================================

data_path = Path(args.data)

if not data_path.exists():
    raise FileNotFoundError(
        f"Training data not found: {data_path}"
    )

print("\nLoading:", data_path)

df = pd.read_csv(data_path)

if "text" not in df.columns:
    raise ValueError(
        "final_data.csv must contain a 'text' column."
    )

df = df.dropna(subset=["text"]).reset_index(drop=True)

print("Training examples:", len(df))


if args.smoke_test:
    df = df.iloc[:16].copy()
    print(
        "SMOKE TEST: using",
        len(df),
        "examples",
    )


dataset = TextPairDataset(df)

collator = FlowVecCollator(
    tokenizer=tokenizer,
    max_length=args.max_length,
)


train_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collator,
    pin_memory=True,
    num_workers=NUM_WORKERS,
    persistent_workers=(NUM_WORKERS > 0),
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adafactor(
    model.parameters(),
    lr=2e-5,
    beta2_decay=-0.8,
    eps=(None, 0.001),
    d=1.0,
    weight_decay=0.0,
)


# Number of optimizer updates
steps_per_epoch = math.ceil(
    len(train_loader) / args.grad_accum
)

total_training_steps = (
    steps_per_epoch * args.epochs
)

warmup_steps = int(
    total_training_steps * WARMUP_RATIO
)


scheduler = get_scheduler(
    name="cosine_with_min_lr",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps,
    scheduler_specific_kwargs={
        "min_lr": MIN_LR
    },
)


# ============================================================
# OUTPUT
# ============================================================

output_dir = Path(args.output)
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

config = {
    "model": MODEL_ID,
    "dimensions": DIMENSIONS,
    "max_length": args.max_length,
    "batch_size": args.batch_size,
    "gradient_accumulation": args.grad_accum,
    "epochs": args.epochs,
    "learning_rate": args.lr,
    "min_lr": MIN_LR,
    "warmup_ratio": WARMUP_RATIO,
    "temperature": TEMPERATURE,
    "seed": SEED,
    "pooling": "last-token",
    "training": "full-parameter",
}

with open(output_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print("Dimensions:", DIMENSIONS)
print("Epochs:", args.epochs)
print("Batch:", args.batch_size)
print("Gradient accumulation:", args.grad_accum)
print("Effective batch:", args.batch_size * args.grad_accum)
print("Temperature:", TEMPERATURE)
print("LR:", args.lr)
print("Warmup steps:", warmup_steps)
print("Total optimizer steps:", total_training_steps)


optimizer.zero_grad(
    set_to_none=True
)


for epoch in range(args.epochs):

    model.train()

    epoch_loss = 0.0
    optimizer_steps = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{args.epochs}",
    )

    for step, batch in enumerate(pbar):

        # ----------------------------------------------------
        # Move inputs to GPU
        # ----------------------------------------------------

        input_ids1 = batch[
            "input_ids1"
        ].to(
            device,
            non_blocking=True,
        )

        attention_mask1 = batch[
            "attention_mask1"
        ].to(
            device,
            non_blocking=True,
        )

        input_ids2 = batch[
            "input_ids2"
        ].to(
            device,
            non_blocking=True,
        )

        attention_mask2 = batch[
            "attention_mask2"
        ].to(
            device,
            non_blocking=True,
        )


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        # L40S supports BF16 natively.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            out1 = model(
                input_ids=input_ids1,
                attention_mask=attention_mask1,
                return_dict=True,
            )

            out2 = model(
                input_ids=input_ids2,
                attention_mask=attention_mask2,
                return_dict=True,
            )

            # FlowVec last-token pooling
            emb1 = last_token_pool(
                out1.last_hidden_state,
                attention_mask1,
            )

            emb2 = last_token_pool(
                out2.last_hidden_state,
                attention_mask2,
            )

            # ------------------------------------------------
            # MRL InfoNCE
            # ------------------------------------------------

            loss = Matry_infonce(
                emb1,
                emb2,
                temperature=TEMPERATURE,
                nested_dims=DIMENSIONS,
            )
            if step < 3:
                print(
                    f"\nDEBUG step={step}"
                    f" | loss={loss.item():.10f}"
                    f" | emb1_mean={emb1.float().mean().item():.6f}"
                    f" | emb2_mean={emb2.float().mean().item():.6f}"
                    f" | emb1_norm={emb1.float().norm(dim=1).mean().item():.6f}"
                    f" | emb2_norm={emb2.float().norm(dim=1).mean().item():.6f}"
                )

            # Gradient accumulation
            loss_for_backward = (
                loss / args.grad_accum
            )


        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss_for_backward.backward()


        # ----------------------------------------------------
        # Optimizer update
        # ----------------------------------------------------

        is_update_step = (
            (step + 1) % args.grad_accum == 0
            or (step + 1) == len(train_loader)
        )

        if is_update_step:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            scheduler.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            optimizer_steps += 1


        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        batch_size = input_ids1.size(0)

        epoch_loss += (
            loss.detach().float().item()
            * batch_size
        )

        avg_loss = epoch_loss / (
            (step + 1) * batch_size
        )

        current_lr = scheduler.get_last_lr()[0]

        gpu_memory = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        gpu_reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        pbar.set_postfix({
            "loss": f"{loss.item():.8f}",
            "avg": f"{avg_loss:.8f}",
            "lr": f"{current_lr:.6e}",
            "gpu": f"{gpu_memory:.1f}/{gpu_reserved:.1f}G",
        })


        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        del (
            input_ids1,
            input_ids2,
            attention_mask1,
            attention_mask2,
            out1,
            out2,
            emb1,
            emb2,
            loss,
            loss_for_backward,
        )


    # ========================================================
    # EPOCH SUMMARY
    # ========================================================

    avg_epoch_loss = epoch_loss / len(dataset)

    print(
        f"\nEpoch {epoch + 1} complete"
    )

    print(
        f"Average loss: {avg_epoch_loss:.6f}"
    )

    print(
        f"Optimizer steps: {optimizer_steps}"
    )

    print(
        f"Learning rate: "
        f"{scheduler.get_last_lr()[0]:.6e}"
    )


    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    if SAVE_EVERY_EPOCH:

        checkpoint_path = (
            output_dir
            / f"flowvec_mrl_epoch_{epoch + 1}.pt"
        )

        print(
            "Saving checkpoint:",
            checkpoint_path,
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_epoch_loss,
                "dimensions": DIMENSIONS,
                "temperature": TEMPERATURE,
            },
            checkpoint_path,
        )

        print("Checkpoint saved.")


    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# FINAL MODEL
# ============================================================

final_path = (
    output_dir
    / "FlowVec_MRL_final.pt"
)

print(
    "\nSaving final model:",
    final_path,
)

torch.save(
    model.state_dict(),
    final_path,
)

print(
    "Final model saved."
)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)