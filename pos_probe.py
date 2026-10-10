"""Compare a POS prefix probe with a learned, fixed-budget dimension selector.

The MIPIC encoder is frozen. The learned probe has exactly ``k`` active
dimensions in every forward pass; only its selector and linear POS head train.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from models.mipic import TinyBERT_MIPIC
from models.tinybert import load_tinybert_tokenizer


UPOS_TAGS = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
)
TAG_TO_ID = {tag: index for index, tag in enumerate(UPOS_TAGS)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TokenBatcher:
    """Tokenize UD words and label only each word's first subword."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: Sequence[dict]) -> dict:
        encoded = self.tokenizer(
            [row["tokens"] for row in rows],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        labels = torch.full_like(encoded["input_ids"], -100)
        for batch_index, row in enumerate(rows):
            previous_word_id = None
            for token_index, word_id in enumerate(encoded.word_ids(batch_index=batch_index)):
                if word_id is not None and word_id != previous_word_id:
                    tag = row["upos"][word_id]
                    if tag not in TAG_TO_ID:
                        raise ValueError(f"Unexpected UD UPOS tag: {tag!r}")
                    labels[batch_index, token_index] = TAG_TO_ID[tag]
                previous_word_id = word_id
        return {**encoded, "labels": labels}


def extract_token_vectors(encoder, tokenizer, split, device, batch_size, max_length):
    loader = DataLoader(
        split,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TokenBatcher(tokenizer, max_length),
        num_workers=0,
    )
    vector_parts, label_parts = [], []
    encoder.eval()
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            output = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            keep = batch["labels"].ne(-100)
            vector_parts.append(output.last_hidden_state[keep.to(device)].float().cpu())
            label_parts.append(batch["labels"][keep].cpu())
    return torch.cat(vector_parts), torch.cat(label_parts)


class PrefixPOSProbe(nn.Module):
    def __init__(self, dimension: int, classes: int):
        super().__init__()
        self.dimension = dimension
        self.classifier = nn.Linear(dimension, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x[:, : self.dimension])


class LearnedTopKPOSProbe(nn.Module):
    """A global dimension selector with an exact hard top-k forward mask.

    The soft top-k relaxation supplies gradients to selector scores, while the
    straight-through mask's forward value is binary and contains exactly k
    ones. The classifier therefore receives only k active coordinates.
    """

    def __init__(self, dimension: int, classes: int, k: int, temperature: float = 0.5):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.scores = nn.Parameter(torch.randn(dimension) * 0.01)
        self.classifier = nn.Linear(dimension, classes)

    def _hard_mask(self) -> torch.Tensor:
        indices = torch.topk(self.scores, self.k).indices
        mask = torch.zeros_like(self.scores)
        return mask.scatter(0, indices, 1.0)

    def _soft_topk_mask(self) -> torch.Tensor:
        # Center a smooth sigmoid surrogate at the current kth-largest score.
        # The threshold is detached; gradients still flow through each score.
        threshold = torch.topk(self.scores.detach(), self.k).values[-1]
        return torch.sigmoid((self.scores - threshold) / self.temperature)

    def selected_mask(self, straight_through: bool) -> torch.Tensor:
        hard = self._hard_mask()
        if not straight_through:
            return hard
        soft = self._soft_topk_mask()
        return hard + soft - soft.detach()

    def forward(self, x: torch.Tensor, *, straight_through: bool = False) -> torch.Tensor:
        mask = self.selected_mask(straight_through)
        return self.classifier(x * mask)


def token_batches(x, y, batch_size, shuffle):
    order = torch.randperm(len(y)) if shuffle else torch.arange(len(y))
    for indices in order.split(batch_size):
        yield x[indices], y[indices]


def evaluate(model, x, y, device, batch_size: int = 8192) -> dict:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(y), batch_size):
            logits = model(x[start : start + batch_size].to(device))
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    truth = y.tolist()
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
    }


def fit_probe(
    model,
    x_train,
    y_train,
    x_dev,
    y_dev,
    device,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best_state = None
    best_dev_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in token_batches(x_train, y_train, batch_size, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, LearnedTopKPOSProbe):
                logits = model(xb, straight_through=True)
            else:
                logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

        dev_metrics = evaluate(model, x_dev, y_dev, device)
        print(
            f"  epoch {epoch}/{epochs}: validation macro-F1 "
            f"{dev_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if dev_metrics["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["macro_f1"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_dev_f1


def pairwise_jaccard(index_sets: List[List[int]]) -> float:
    sets = [set(indices) for indices in index_sets]
    scores = [
        len(left & right) / len(left | right)
        for i, left in enumerate(sets)
        for right in sets[i + 1 :]
    ]
    return float(np.mean(scores)) if scores else 1.0


def mean_and_std(rows: List[dict], key: str) -> dict:
    values = [row[key] for row in rows]
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("pos_probe_budget_results"))
    parser.add_argument("--k-values", default="16,32,64,128,256")
    parser.add_argument("--seeds", default="13,37,73")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"MIPIC checkpoint not found: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(value) for value in args.seeds.split(",")]
    print(f"Device: {device}; checkpoint: {args.checkpoint}", flush=True)

    tokenizer = load_tinybert_tokenizer()
    model = TinyBERT_MIPIC(device=device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    encoder = model.encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    dimension = int(encoder.config.hidden_size)
    k_values = sorted({int(value) for value in args.k_values.split(",")})
    if not k_values or min(k_values) < 1 or max(k_values) > dimension:
        raise ValueError(f"k-values must be between 1 and {dimension}, got {k_values}")

    dataset = load_dataset(
        "universal-dependencies/universal_dependencies",
        "en_ewt",
        revision="2.18",
    )
    print("Extracting frozen final-layer token vectors from UD English EWT...", flush=True)
    features, labels = {}, {}
    for name, split_name in (("train", "train"), ("validation", "dev"), ("test", "test")):
        features[name], labels[name] = extract_token_vectors(
            encoder,
            tokenizer,
            dataset[split_name],
            device,
            args.encoder_batch_size,
            args.max_length,
        )
        print(f"  {name}: {len(labels[name]):,} tagged tokens", flush=True)

    del model, encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    x_train, y_train = features["train"], labels["train"]
    x_dev, y_dev = features["validation"], labels["validation"]
    x_test, y_test = features["test"], labels["test"]

    # Apply the same train-only normalization to both conditions. This keeps
    # activation scale from determining selector rank and avoids test leakage.
    mean = x_train.mean(dim=0)
    scale = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    for x in (x_train, x_dev, x_test):
        x.sub_(mean).div_(scale)

    results = []
    selected_by_seed = {}
    for seed in seeds:
        selected_by_seed[str(seed)] = {}
        for k in k_values:
            print(f"\nSeed {seed}: prefix-{k}", flush=True)
            seed_everything(seed)
            prefix = PrefixPOSProbe(k, len(UPOS_TAGS))
            prefix, prefix_dev_f1 = fit_probe(
                prefix,
                x_train,
                y_train,
                x_dev,
                y_dev,
                device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
            prefix_test = evaluate(prefix, x_test, y_test, device)

            print(f"Seed {seed}: learned exact-{k}", flush=True)
            seed_everything(seed)
            learned = LearnedTopKPOSProbe(dimension, len(UPOS_TAGS), k)
            learned, learned_dev_f1 = fit_probe(
                learned,
                x_train,
                y_train,
                x_dev,
                y_dev,
                device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
            selected = torch.where(learned.selected_mask(straight_through=False).cpu() > 0)[0].tolist()
            if len(selected) != k:
                raise RuntimeError(f"Selector activated {len(selected)} dimensions, expected exactly {k}")
            learned_test = evaluate(learned, x_test, y_test, device)
            fraction_in_prefix = sum(index < k for index in selected) / k

            row = {
                "seed": seed,
                "k": k,
                "prefix_dev_macro_f1": prefix_dev_f1,
                "prefix_test_accuracy": prefix_test["accuracy"],
                "prefix_test_macro_f1": prefix_test["macro_f1"],
                "learned_dev_macro_f1": learned_dev_f1,
                "learned_test_accuracy": learned_test["accuracy"],
                "learned_test_macro_f1": learned_test["macro_f1"],
                "selected_fraction_in_first_k": fraction_in_prefix,
                "active_dimension_count": len(selected),
                "selected_indices": selected,
            }
            results.append(row)
            selected_by_seed[str(seed)][str(k)] = selected
            print(
                f"seed={seed} k={k}: prefix F1={prefix_test['macro_f1']:.4f}; "
                f"learned exact-k F1={learned_test['macro_f1']:.4f}; "
                f"selected first-k fraction={fraction_in_prefix:.2f}",
                flush=True,
            )

    summary = {}
    for k in k_values:
        rows_at_k = [row for row in results if row["k"] == k]
        indices_at_k = [selected_by_seed[str(seed)][str(k)] for seed in seeds]
        summary[str(k)] = {
            "prefix_test_accuracy": mean_and_std(rows_at_k, "prefix_test_accuracy"),
            "prefix_test_macro_f1": mean_and_std(rows_at_k, "prefix_test_macro_f1"),
            "learned_test_accuracy": mean_and_std(rows_at_k, "learned_test_accuracy"),
            "learned_test_macro_f1": mean_and_std(rows_at_k, "learned_test_macro_f1"),
            "selected_fraction_in_first_k": mean_and_std(rows_at_k, "selected_fraction_in_first_k"),
            "mean_pairwise_jaccard": pairwise_jaccard(indices_at_k),
        }
    report = {
        "dataset": "Universal Dependencies English EWT",
        "representation": "frozen MIPIC final-layer first-subword token vectors",
        "checkpoint": str(args.checkpoint),
        "dimension": dimension,
        "k_values": k_values,
        "selection_method": "straight-through differentiable top-k; exactly k active coordinates per forward pass",
        "normalization": "per-coordinate mean and standard deviation from train tokens only; shared by both conditions",
        "seeds": seeds,
        "metrics": results,
        "summary": summary,
        "selected_indices_by_seed": selected_by_seed,
        "mean_pairwise_jaccard_by_k": {
            k: summary[k]["mean_pairwise_jaccard"] for k in summary
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pos_probe_budget_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(results).drop(columns="selected_indices").to_csv(
        args.output_dir / "pos_probe_budget_results.csv", index=False
    )
    print(f"Saved fixed-budget POS probe results to {args.output_dir}", flush=True)
    print("Summary (mean ± std across seeds):", flush=True)
    for k, metrics in summary.items():
        print(f"  k={k}", flush=True)
        for name, values in metrics.items():
            if isinstance(values, dict):
                print(f"    {name}: {values['mean']:.4f} ± {values['std']:.4f}", flush=True)
            else:
                print(f"    {name}: {values:.4f}", flush=True)


if __name__ == "__main__":
    main()
