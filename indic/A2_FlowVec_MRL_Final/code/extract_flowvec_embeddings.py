import os
import argparse
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "LingoIITGN/FlowVec-v1"

MAX_LENGTH = 256
BATCH_SIZE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FlowVec uses 4096-dimensional embeddings
EMBEDDING_DIM = 4096


# ============================================================
# LAST-TOKEN POOLING
# ============================================================

def last_token_pool(
    last_hidden_states,
    attention_mask,
):
    """
    FlowVec last-token pooling.

    For right-padded sequences, the last real token is the
    position immediately before the padding starts.
    """

    sequence_lengths = attention_mask.sum(dim=1) - 1

    batch_indices = torch.arange(
        last_hidden_states.size(0),
        device=last_hidden_states.device,
    )

    embeddings = last_hidden_states[
        batch_indices,
        sequence_lengths,
    ]

    return embeddings


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(checkpoint_path):

    print("=" * 70)
    print("Loading FlowVec model")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # --------------------------------------------------------
    # Load our fine-tuned checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    print(f"Checkpoint type: {type(checkpoint)}")

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected state_dict, got {type(checkpoint)}"
        )

    missing, unexpected = model.load_state_dict(
        checkpoint,
        strict=True,
    )

    print("Checkpoint loaded successfully.")
    print(f"Missing keys   : {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("WARNING: Missing keys detected!")
        print(missing[:10])

    if len(unexpected) > 0:
        print("WARNING: Unexpected keys detected!")
        print(unexpected[:10])

    model = model.to(DEVICE)
    model.eval()

    print(f"Device: {DEVICE}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")

    return tokenizer, model


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

@torch.no_grad()
def extract_embeddings(
    texts,
    tokenizer,
    model,
    batch_size=BATCH_SIZE,
):

    all_embeddings = []

    print(f"\nExtracting {len(texts)} embeddings...")
    print(f"Batch size: {batch_size}")

    for start in range(0, len(texts), batch_size):

        end = min(
            start + batch_size,
            len(texts),
        )

        batch_texts = texts[start:end]

        # ----------------------------------------------------
        # Tokenization
        # ----------------------------------------------------

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            k: v.to(DEVICE)
            for k, v in encoded.items()
        }

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            **encoded,
            return_dict=True,
        )

        # Last hidden state:
        # [batch, sequence_length, 4096]

        last_hidden_states = outputs.last_hidden_state

        # ----------------------------------------------------
        # Last-token pooling
        # ----------------------------------------------------

        embeddings = last_token_pool(
            last_hidden_states,
            encoded["attention_mask"],
        )

        # ----------------------------------------------------
        # L2 normalization
        # ----------------------------------------------------

        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        # ----------------------------------------------------
        # Move to CPU
        # ----------------------------------------------------

        all_embeddings.append(
            embeddings.float().cpu().numpy()
        )

        if (start // batch_size) % 50 == 0:
            print(
                f"Processed {end}/{len(texts)} "
                f"({100 * end / len(texts):.2f}%)"
            )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    return embeddings


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to FlowVec_MRL_final.pt",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="CSV containing the text column",
    )

    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of text column",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npy file",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(args.input)

    if args.text_column not in df.columns:
        raise ValueError(
            f"Column '{args.text_column}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    texts = (
        df[args.text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(f"Loaded {len(texts)} texts")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    tokenizer, model = load_model(
        args.checkpoint
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    embeddings = extract_embeddings(
        texts,
        tokenizer,
        model,
        batch_size=args.batch_size,
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Embedding extraction complete")
    print("=" * 70)

    print(f"Shape : {embeddings.shape}")
    print(f"Dtype : {embeddings.dtype}")

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    print(
        f"Norm  : mean={norms.mean():.6f}, "
        f"min={norms.min():.6f}, "
        f"max={norms.max():.6f}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(args.output)
        if os.path.dirname(args.output)
        else ".",
        exist_ok=True,
    )

    np.save(
        args.output,
        embeddings,
    )

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()