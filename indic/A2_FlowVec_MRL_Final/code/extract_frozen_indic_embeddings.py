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

LANGUAGES = ["hi", "bn", "gu", "ta", "te"]

DATA_DIR = Path("data/indic")

# IMPORTANT:
# Separate directory from MRL embeddings so nothing gets overwritten.
OUTPUT_DIR = Path("results/flowvec_frozen/indic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 256
BATCH_SIZE = 8


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Frozen FlowVec-v1 Indic Embedding Extraction")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Load tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

# FlowVec uses right padding
tokenizer.padding_side = "right"

print("Padding side:", tokenizer.padding_side)
print("Pad token:", tokenizer.pad_token)
print("EOS token:", tokenizer.eos_token)


# ============================================================
# Load ORIGINAL FROZEN FlowVec-v1
# ============================================================

print("\nLoading original frozen FlowVec-v1...")

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model.to(device)
model.eval()

# Make absolutely sure no gradients are computed
for param in model.parameters():
    param.requires_grad = False

print("Model loaded.")
print("Parameters are frozen.")


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

    # Forward pass
    outputs = model(**encoded)

    # FlowVec-v1 uses last-token pooling
    hidden_states = outputs.last_hidden_state

    attention_mask = encoded["attention_mask"]

    # Index of the final non-padding token
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

    # Convert to float32 before normalization
    embeddings = embeddings.float()

    # L2 normalize
    embeddings = F.normalize(
        embeddings,
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
        f"{lang}_{split}_frozen.npy"
    )

    print("\n" + "=" * 60)
    print(f"Language: {lang}")
    print(f"Split: {split}")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(input_file)

    texts = df["text"].tolist()

    print("Input:", input_file)
    print("Examples:", len(texts))

    # --------------------------------------------------------
    # Extract embeddings
    # --------------------------------------------------------

    all_embeddings = []

    for start in tqdm(
        range(0, len(texts), BATCH_SIZE),
        desc=f"Extracting {lang} {split}",
    ):

        batch_texts = texts[
            start:start + BATCH_SIZE
        ]

        embeddings = encode_texts(
            batch_texts
        )

        all_embeddings.append(embeddings)

    # --------------------------------------------------------
    # Combine batches
    # --------------------------------------------------------

    embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    print("\nShape:", embeddings.shape)
    print("Dtype:", embeddings.dtype)

    # --------------------------------------------------------
    # Verify normalization
    # --------------------------------------------------------

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    print(
        "Norm:",
        "mean =", norms.mean(),
        "min =", norms.min(),
        "max =", norms.max(),
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        output_file,
        embeddings.astype(np.float32),
    )

    print("Saved ->", output_file)


# ============================================================
# Main
# ============================================================

for lang in LANGUAGES:

    extract_split(
        lang,
        "train",
    )

    extract_split(
        lang,
        "test",
    )


# ============================================================
# Final verification
# ============================================================

print("\n" + "=" * 60)
print("Extraction complete!")
print("=" * 60)

for lang in LANGUAGES:

    train_file = (
        OUTPUT_DIR /
        f"{lang}_train_frozen.npy"
    )

    test_file = (
        OUTPUT_DIR /
        f"{lang}_test_frozen.npy"
    )

    train_embeddings = np.load(
        train_file
    )

    test_embeddings = np.load(
        test_file
    )

    print(
        f"{lang}: "
        f"train={train_embeddings.shape}, "
        f"test={test_embeddings.shape}"
    )

print("\nAll frozen Indic embeddings extracted successfully.")