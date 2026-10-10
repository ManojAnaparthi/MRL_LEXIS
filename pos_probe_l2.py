"""Dense POS probe with trainable per-coordinate gates and L2 regularization.

The MIPIC encoder is frozen. Every coordinate remains present; one sigmoid
gate scales each coordinate, and an L2 penalty encourages smaller gate values.
The experiment reports the learned gates by dimension rather than selecting a
fixed number of active dimensions.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr
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


class DensePOSProbe(nn.Module):
    """Full-width linear POS probe; optionally learn a sigmoid per-coordinate gate."""

    def __init__(self, dimension: int, classes: int, gated: bool):
        super().__init__()
        self.gated = gated
        if gated:
            # Start with nearly unscaled coordinates; training can reduce or
            # increase each value in (0, 1), while L2 favors smaller values.
            initial_gate = 0.9
            initial_logit = float(np.log(initial_gate / (1.0 - initial_gate)))
            self.gate_logits = nn.Parameter(torch.full((dimension,), initial_logit))
        self.classifier = nn.Linear(dimension, classes)

    def gates(self) -> torch.Tensor:
        if not self.gated:
            return torch.ones(self.classifier.in_features, device=self.classifier.weight.device)
        return torch.sigmoid(self.gate_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x * self.gates())


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
    model, x_train, y_train, x_dev, y_dev, device, *, epochs, batch_size,
    learning_rate, gate_l2,
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
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            if model.gated:
                # L2 penalty on the dense gate vector. This encourages smaller
                # scales but does not enforce a threshold or sparsity budget.
                loss = loss + gate_l2 * model.gates().pow(2).sum()
            loss.backward()
            optimizer.step()

        dev_metrics = evaluate(model, x_dev, y_dev, device)
        print(
            f"  epoch {epoch}/{epochs}: validation macro-F1 "
            f"{dev_metrics['macro_f1']:.4f}", flush=True,
        )
        if dev_metrics["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["macro_f1"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    return model, best_dev_f1


def mean_std(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("pos_probe_l2_results"))
    parser.add_argument("--seeds", default="13,37,73")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--gate-l2", type=float, default=1e-3)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"MIPIC checkpoint not found: {args.checkpoint}")
    if args.gate_l2 < 0:
        raise ValueError("--gate-l2 must be non-negative")

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

    dataset = load_dataset(
        "universal-dependencies/universal_dependencies",
        "en_ewt",
        revision="2.18",
    )
    print("Extracting frozen final-layer token vectors from UD English EWT...", flush=True)
    features, labels = {}, {}
    for name, split_name in (("train", "train"), ("validation", "dev"), ("test", "test")):
        features[name], labels[name] = extract_token_vectors(
            encoder, tokenizer, dataset[split_name], device,
            args.encoder_batch_size, args.max_length,
        )
        print(f"  {name}: {len(labels[name]):,} tagged tokens", flush=True)
    del model, encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    x_train, y_train = features["train"], labels["train"]
    x_dev, y_dev = features["validation"], labels["validation"]
    x_test, y_test = features["test"], labels["test"]
    # Train-only normalization avoids using validation/test distribution stats.
    mean = x_train.mean(dim=0)
    scale = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    for x in (x_train, x_dev, x_test):
        x.sub_(mean).div_(scale)

    rows = []
    gates_by_seed = {}
    for seed in seeds:
        print(f"\nSeed {seed}: full-width baseline", flush=True)
        seed_everything(seed)
        baseline = DensePOSProbe(dimension, len(UPOS_TAGS), gated=False)
        baseline, baseline_dev_f1 = fit_probe(
            baseline, x_train, y_train, x_dev, y_dev, device,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, gate_l2=0.0,
        )
        baseline_test = evaluate(baseline, x_test, y_test, device)

        print(f"Seed {seed}: dense learned gates (L2={args.gate_l2:g})", flush=True)
        seed_everything(seed)
        gated = DensePOSProbe(dimension, len(UPOS_TAGS), gated=True)
        gated, gated_dev_f1 = fit_probe(
            gated, x_train, y_train, x_dev, y_dev, device,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, gate_l2=args.gate_l2,
        )
        gated_test = evaluate(gated, x_test, y_test, device)
        gate_values = gated.gates().detach().cpu().numpy()
        gates_by_seed[str(seed)] = gate_values.tolist()
        prefix_count = min(128, dimension)
        remaining = gate_values[prefix_count:]
        rho, rho_p = spearmanr(np.arange(dimension), gate_values)
        rows.append({
            "seed": seed,
            "baseline_dev_macro_f1": baseline_dev_f1,
            "baseline_test_accuracy": baseline_test["accuracy"],
            "baseline_test_macro_f1": baseline_test["macro_f1"],
            "gated_dev_macro_f1": gated_dev_f1,
            "gated_test_accuracy": gated_test["accuracy"],
            "gated_test_macro_f1": gated_test["macro_f1"],
            "gate_mean": float(gate_values.mean()),
            "gate_std": float(gate_values.std()),
            "gate_min": float(gate_values.min()),
            "gate_max": float(gate_values.max()),
            "fraction_gate_below_0_5": float(np.mean(gate_values < 0.5)),
            "fraction_gate_below_0_25": float(np.mean(gate_values < 0.25)),
            "first_128_gate_mean": float(gate_values[:prefix_count].mean()),
            "remaining_gate_mean": float(remaining.mean()) if len(remaining) else None,
            "dimension_index_spearman_rho": float(rho),
            "dimension_index_spearman_p": float(rho_p),
        })
        print(
            f"seed={seed}: baseline F1={baseline_test['macro_f1']:.4f}; "
            f"gated F1={gated_test['macro_f1']:.4f}; "
            f"mean gate={gate_values.mean():.3f}; "
            f"first-{prefix_count} mean={gate_values[:prefix_count].mean():.3f}; "
            f"Spearman(index, gate)={rho:.3f}", flush=True,
        )

    summary = {}
    numeric_keys = [key for key in rows[0] if key != "seed" and key != "remaining_gate_mean"]
    for key in numeric_keys:
        summary[key] = mean_std([float(row[key]) for row in rows])
    if dimension > 128:
        summary["remaining_gate_mean"] = mean_std(
            [float(row["remaining_gate_mean"]) for row in rows]
        )

    gate_matrix = np.asarray([gates_by_seed[str(seed)] for seed in seeds])
    prefix_sizes = sorted({size for size in (32, 64, 128, 256, 512, dimension) if size <= dimension})
    prefix_gate_means_by_seed = {
        str(seed): {
            str(size): float(np.mean(gates_by_seed[str(seed)][:size]))
            for size in prefix_sizes
        }
        for seed in seeds
    }
    prefix_gate_summary = {
        str(size): mean_std([
            prefix_gate_means_by_seed[str(seed)][str(size)] for seed in seeds
        ])
        for size in prefix_sizes
    }
    dimension_rows = []
    for index in range(dimension):
        dimension_rows.append({
            "dimension": index,
            "gate_mean": float(gate_matrix[:, index].mean()),
            "gate_std_across_seeds": float(gate_matrix[:, index].std()),
            **{f"gate_seed_{seed}": float(gates_by_seed[str(seed)][index]) for seed in seeds},
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "Universal Dependencies English EWT",
        "representation": "frozen MIPIC final-layer first-subword token vectors",
        "checkpoint": str(args.checkpoint),
        "dimension": dimension,
        "seeds": seeds,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gate_l2": args.gate_l2,
        "gate_penalty": "gate_l2 * sum(sigmoid(gate_logits)^2)",
        "normalization": "per-coordinate mean and standard deviation from train tokens only",
        "selection_method": "dense learned sigmoid coordinate gates; no active-dimension budget",
        "metrics_by_seed": rows,
        "summary_mean_std": summary,
        "prefix_gate_means_by_seed": prefix_gate_means_by_seed,
        "prefix_gate_mean_summary": prefix_gate_summary,
        "gates_by_seed": gates_by_seed,
    }
    (args.output_dir / "pos_probe_l2_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(args.output_dir / "pos_probe_l2_summary.csv", index=False)
    pd.DataFrame(dimension_rows).to_csv(
        args.output_dir / "pos_probe_l2_dimension_gates.csv", index=False
    )
    prefix_rows = []
    for size in prefix_sizes:
        row = {"prefix_dimensions": size}
        for seed in seeds:
            row[f"seed_{seed}_mean_gate"] = prefix_gate_means_by_seed[str(seed)][str(size)]
        row["mean_gate_across_seeds"] = prefix_gate_summary[str(size)]["mean"]
        row["std_across_seeds"] = prefix_gate_summary[str(size)]["std"]
        prefix_rows.append(row)
    pd.DataFrame(prefix_rows).to_csv(
        args.output_dir / "pos_probe_l2_prefix_gate_means.csv", index=False
    )
    print(f"Saved L2-gated POS probe results to {args.output_dir}", flush=True)
    print("Summary (mean ± population std across seeds):", flush=True)
    for key, values in summary.items():
        print(f"  {key}: {values['mean']:.4f} ± {values['std']:.4f}", flush=True)
    print("Mean gate value over each initial prefix (mean ± std across seeds):", flush=True)
    for size, values in prefix_gate_summary.items():
        print(f"  first {size}: {values['mean']:.4f} ± {values['std']:.4f}", flush=True)


if __name__ == "__main__":
    main()
