import argparse
import numpy as np
import torch

from common import (
    DIMENSIONS,
    FLOWVEC_DIM,
    MODEL_NAME,
    encode_and_cache,
    get_device,
    load_flowvec,
    set_seed,
)

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--cache", default="embeddings/smoke_flowvec.npz")
args = parser.parse_args()

set_seed()

print("=== FlowVec-v1 smoke test ===")
print("Model:", MODEL_NAME)
print("Device:", get_device())

if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. This test will run on CPU.")

model = load_flowvec()

# Keep this deliberately tiny: it verifies model loading and embedding shape.
texts = [
    "This is a short English sentence.",
    "FlowVec should produce a fixed-size embedding.",
    "यह एक हिंदी परीक्षण वाक्य है।",
    "இது ஒரு தமிழ் சோதனை வாக்கியம்.",
]

embeddings = encode_and_cache(
    model,
    texts,
    args.cache,
    batch_size=args.batch_size,
)

print("Embedding shape:", embeddings.shape)
print("dtype:", embeddings.dtype)
print("finite:", np.isfinite(embeddings).all())

assert embeddings.shape == (len(texts), FLOWVEC_DIM)
assert np.isfinite(embeddings).all()

for dimension in DIMENSIONS:
    prefix = embeddings[:, :dimension]
    assert prefix.shape == (len(texts), dimension)

print("All prefix dimensions:", DIMENSIONS)
print("\nFLOWVEC SMOKE TEST PASSED")
