import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "LingoIITGN/FlowVec-v1"

CHECKPOINT = (
    "results/flowvec_mrl_indic/FlowVec_MRL_Indic_final.pt"
)

LANGUAGES = ["hi", "bn", "gu", "ta", "te"]

DATA_DIR = Path("data/indic")
OUTPUT_DIR = Path("results/flowvec_mrl_indic/indic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 256
BATCH_SIZE = 8


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Load tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

tokenizer.padding_side = "right"


# ============================================================
# Load model
# ============================================================

print("Loading FlowVec-v1...")

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Load English-MRL fine-tuned weights
print("Loading MRL checkpoint...")

state_dict = torch.load(
    CHECKPOINT,
    map_location="cpu",
)

model.load_state_dict(
    state_dict,
    strict=True,
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# Last-token pooling
# ============================================================

@torch.no_grad()
def encode_texts(texts):

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    outputs = model(**encoded)

    # FlowVec uses last-token pooling
    hidden_states = outputs.last_hidden_state

    attention_mask = encoded["attention_mask"]

    last_token_indices = (
        attention_mask.sum(dim=1) - 1
    )

    batch_indices = torch.arange(
        hidden_states.size(0),
        device=device,
    )

    embeddings = hidden_states[
        batch_indices,
        last_token_indices,
    ]

    # L2 normalization
    embeddings = F.normalize(
        embeddings.float(),
        p=2,
        dim=1,
    )

    return embeddings.cpu().numpy()


# ============================================================
# Extract one split
# ============================================================

def extract_split(lang, split):

    input_file = DATA_DIR / f"{lang}_{split}.csv"

    output_file = (
        OUTPUT_DIR /
        f"{lang}_{split}_mrl.npy"
    )

    df = pd.read_csv(input_file)

    texts = df["text"].tolist()

    all_embeddings = []

    print(
        f"\nExtracting {lang} {split}: "
        f"{len(texts)} examples"
    )

    for start in tqdm(
        range(0, len(texts), BATCH_SIZE)
    ):

        batch_texts = texts[
            start:start + BATCH_SIZE
        ]

        embeddings = encode_texts(
            batch_texts
        )

        all_embeddings.append(embeddings)

    embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    print("Shape:", embeddings.shape)
    print("Dtype:", embeddings.dtype)

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    print(
        "Norm:",
        norms.mean(),
        norms.min(),
        norms.max(),
    )

    np.save(
        output_file,
        embeddings.astype(np.float32),
    )

    print(
        f"Saved -> {output_file}"
    )


# ============================================================
# Main
# ============================================================

for lang in LANGUAGES:

    extract_split(lang, "train")
    extract_split(lang, "test")


print("\nAll Indic embeddings extracted.")