#!/usr/bin/env python3

import os
import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

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

DATA_PATH = "data/indic_mrl/indic_mrl_train.csv"
OUTPUT_DIR = "results/flowvec_mrl_indic"

# Matryoshka dimensions
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

# Same configuration as successful English MRL run
MAX_LENGTH = 256
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1

EPOCHS = 5

LR = 2e-5
MIN_LR = 2e-6
WARMUP_RATIO = 0.10

TEMPERATURE = 0.05

SEED = 42

NUM_WORKERS = 2

SAVE_EVERY_EPOCH = True


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
print("FLOWVEC A2 INDIC MRL TRAINING")
print("=" * 70)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

print(
    "GPU memory:",
    round(
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3,
        2,
    ),
    "GB",
)

device = torch.device("cuda")


# ============================================================
# DATASET
# ============================================================

class TextPairDataset(Dataset):

    def __init__(self, dataframe):
        self.texts = (
            dataframe["text"]
            .astype(str)
            .tolist()
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]

        # Same-sentence positive pair exactly as A1.
        # Two stochastic forward passes through dropout
        # provide the two views.
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
# FLOWVEC LAST-TOKEN POOLING
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

        # Normalize each truncated representation
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # In-batch similarity
        logits = torch.matmul(q, k.T)

        logits = logits / temperature

        # Positive pair is diagonal
        labels = torch.arange(
            q.size(0),
            device=q.device,
        )

        # Direction A -> B
        loss_ab = F.cross_entropy(
            logits,
            labels,
        )

        # Direction B -> A
        loss_ba = F.cross_entropy(
            logits.T,
            labels,
        )

        # Symmetric InfoNCE
        loss = 0.5 * (loss_ab + loss_ba)

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

# Ensure right padding, as used by FlowVec
tokenizer.padding_side = "right"

print("Tokenizer loaded.")
print("Padding side:", tokenizer.padding_side)


print("\nLoading FlowVec-v1...")

# BF16 is preferable on L40S
model = AutoModel.from_pretrained(
    MODEL_ID,
    subfolder="0_Transformer",
    dtype=torch.bfloat16,
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

# IMPORTANT:
# This was present in the successful English MRL script.
# It dramatically reduces activation memory during backward.

if hasattr(model, "gradient_checkpointing_enable"):

    print("\nEnabling gradient checkpointing...")

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )

    print("Gradient checkpointing enabled.")

else:

    print(
        "WARNING: Model does not expose "
        "gradient_checkpointing_enable()."
    )


# Disable KV cache during training
if hasattr(model, "config"):

    if hasattr(model.config, "use_cache"):

        model.config.use_cache = False

        print("use_cache = False")


# ============================================================
# TRAINABLE PARAMETERS
# ============================================================

# Full-parameter fine-tuning.
# No LoRA.

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

data_path = Path(DATA_PATH)

if not data_path.exists():
    raise FileNotFoundError(
        f"Training data not found: {data_path}"
    )


print("\nLoading:", data_path)

df = pd.read_csv(data_path)

if "text" not in df.columns:
    raise ValueError(
        "Indic MRL CSV must contain a 'text' column."
    )


df = df.dropna(
    subset=["text"]
).reset_index(drop=True)


print("\nDataset statistics")
print("-" * 50)

print("Total examples:", len(df))


if "language" in df.columns:

    print("\nLanguage distribution:")
    print(df["language"].value_counts())


# ============================================================
# DATASET / DATALOADER
# ============================================================

dataset = TextPairDataset(df)

collator = FlowVecCollator(
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
)


train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
    pin_memory=True,
    num_workers=NUM_WORKERS,
    persistent_workers=(NUM_WORKERS > 0),
    drop_last=True,
)


# ============================================================
# OPTIMIZER
# ============================================================

print("\nCreating Adafactor optimizer...")

optimizer = torch.optim.Adafactor(
    model.parameters(),
    lr=LR,
    beta2_decay=-0.8,
    eps=(None, 0.001),
    d=1.0,
    weight_decay=0.0,
)


# ============================================================
# LR SCHEDULER
# ============================================================

steps_per_epoch = math.ceil(
    len(train_loader) / GRAD_ACCUM_STEPS
)

total_training_steps = (
    steps_per_epoch * EPOCHS
)

warmup_steps = int(
    total_training_steps * WARMUP_RATIO
)


print("Steps per epoch:", steps_per_epoch)
print("Total optimizer steps:", total_training_steps)
print("Warmup steps:", warmup_steps)


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

output_dir = Path(OUTPUT_DIR)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


config = {

    "model": MODEL_ID,

    "dimensions": DIMENSIONS,

    "max_length": MAX_LENGTH,

    "batch_size": BATCH_SIZE,

    "gradient_accumulation":
        GRAD_ACCUM_STEPS,

    "effective_batch_size":
        BATCH_SIZE * GRAD_ACCUM_STEPS,

    "epochs": EPOCHS,

    "learning_rate": LR,

    "min_lr": MIN_LR,

    "warmup_ratio": WARMUP_RATIO,

    "temperature": TEMPERATURE,

    "seed": SEED,

    "pooling": "last-token",

    "training": "full-parameter",

    "optimizer": "Adafactor",

    "gradient_checkpointing": True,

    "use_cache": False,

    "corpus": "AI4Bharat IndicCorpV2",

    "languages": [
        "hi",
        "bn",
        "gu",
        "ta",
        "te",
    ],
}


with open(
    output_dir / "config.json",
    "w",
) as f:

    json.dump(
        config,
        f,
        indent=2,
    )


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print("Dimensions:", DIMENSIONS)
print("Epochs:", EPOCHS)
print("Batch:", BATCH_SIZE)
print(
    "Gradient accumulation:",
    GRAD_ACCUM_STEPS,
)
print(
    "Effective batch:",
    BATCH_SIZE * GRAD_ACCUM_STEPS,
)
print("Temperature:", TEMPERATURE)
print("LR:", LR)
print("Warmup steps:", warmup_steps)
print(
    "Total optimizer steps:",
    total_training_steps,
)


optimizer.zero_grad(
    set_to_none=True
)


# ============================================================
# EPOCH LOOP
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0.0

    optimizer_steps = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
    )


    # --------------------------------------------------------
    # BATCH LOOP
    # --------------------------------------------------------

    for step, batch in enumerate(pbar):

        # ====================================================
        # MOVE INPUTS TO GPU
        # ====================================================

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


        # ====================================================
        # FORWARD
        # ====================================================

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


            # =================================================
            # LAST-TOKEN POOLING
            # =================================================

            emb1 = last_token_pool(
                out1.last_hidden_state,
                attention_mask1,
            )

            emb2 = last_token_pool(
                out2.last_hidden_state,
                attention_mask2,
            )


            # =================================================
            # MRL INFONCE
            # =================================================

            loss = Matry_infonce(
                emb1,
                emb2,
                temperature=TEMPERATURE,
                nested_dims=DIMENSIONS,
            )


            # =================================================
            # GRADIENT ACCUMULATION
            # =================================================

            loss_for_backward = (
                loss / GRAD_ACCUM_STEPS
            )


        # ====================================================
        # BACKWARD
        # ====================================================

        loss_for_backward.backward()


        # ====================================================
        # OPTIMIZER UPDATE
        # ====================================================

        is_update_step = (
            (step + 1) % GRAD_ACCUM_STEPS == 0
            or
            (step + 1) == len(train_loader)
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


        # ====================================================
        # LOGGING
        # ====================================================

        batch_size = input_ids1.size(0)

        epoch_loss += (
            loss.detach().float().item()
            * batch_size
        )

        avg_loss = epoch_loss / (
            (step + 1) * batch_size
        )

        current_lr = (
            scheduler.get_last_lr()[0]
        )


        gpu_memory = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        gpu_reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )


        pbar.set_postfix({

            "loss":
                f"{loss.item():.8f}",

            "avg":
                f"{avg_loss:.8f}",

            "lr":
                f"{current_lr:.6e}",

            "gpu":
                f"{gpu_memory:.1f}/"
                f"{gpu_reserved:.1f}G",

        })


        # ====================================================
        # CLEANUP
        # ====================================================

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

    avg_epoch_loss = (
        epoch_loss / len(dataset)
    )


    print(
        f"\nEpoch {epoch + 1} complete"
    )

    print(
        f"Average loss: "
        f"{avg_epoch_loss:.6f}"
    )

    print(
        f"Optimizer steps: "
        f"{optimizer_steps}"
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
            / f"flowvec_mrl_indic_epoch_{epoch + 1}.pt"
        )


        print(
            "Saving checkpoint:",
            checkpoint_path,
        )


        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "loss":
                    avg_epoch_loss,

                "dimensions":
                    DIMENSIONS,

                "temperature":
                    TEMPERATURE,

            },
            checkpoint_path,
        )


        print("Checkpoint saved.")


    # ========================================================
    # CLEANUP
    # ========================================================

    gc.collect()

    torch.cuda.empty_cache()


# ============================================================
# FINAL MODEL
# ============================================================

final_path = (
    output_dir
    / "FlowVec_MRL_Indic_final.pt"
)


print(
    "\nSaving final model:",
    final_path,
)


torch.save(
    model.state_dict(),
    final_path,
)


print("Final model saved.")


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)