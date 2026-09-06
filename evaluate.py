"""TweetEval evaluation refactored from ``mrl-baseline-a1.ipynb``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from losses.mrl_loss import MRL_NESTED_DIMS
from models.tinybert import mrl_sentence_embedding


@dataclass(frozen=True)
class MRLEvaluationConfig:
    dimensions: Tuple[int, ...] = MRL_NESTED_DIMS
    batch_size: int = 64
    max_length: int = 256
    # None = use the WHOLE probe-train split, which is the standard
    # Conneau & Kiela protocol the paper cites in A.4 and what both the official
    # notebook and the reference MRL baseline do. Table 6's "Train (Sampled)
    # 3,000" is the count sampled into the *contrastive training corpus*
    # (A.2: "6,000 sentences from classification datasets, 3,000 per dataset"),
    # not the probe's training size.
    #
    # Measured on one frozen untrained TinyBERT-6L, macro-F1, probe rows only:
    #     3,000 rows   51.11 / 59.20 / 61.90 / 65.54 / 67.72 / 68.83 / 69.23
    #    26,732 rows   62.62 / 65.90 / 67.85 / 69.64 / 70.73 / 71.67 / 71.99
    # i.e. +11.51 F1 at d=16 and +2.76 at d=768 from probe data alone. A 768-dim
    # logistic regression fitted on 3,000 examples is data-starved.
    train_samples: Optional[int] = None
    random_state: int = 42
    # sklearn's LBFGS default of 200 iterations does not converge on raw
    # (unbounded) [CLS] hidden states at the wider prefixes: n_iter_ pins at the
    # 200 cap for d=768 on a pretrained TinyBERT-6L probe. Allowing the same
    # solver on the same features to actually reach its optimum is a solver
    # setting, not a protocol change -- it is applied identically to every model
    # and every dimension, and ``converged`` is reported per dimension.
    max_iter: int = 1000
    # Optional cap on the *test* split, used only to validate the end-to-end
    # train -> checkpoint -> evaluate pipeline cheaply. Leave at None for any
    # reported number; the default protocol uses the full TweetEval test set.
    test_samples: Optional[int] = None


class ClassificationDataset(Dataset[Tuple[str, int]]):
    def __init__(
        self,
        csv_path: str | Path,
        max_samples: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        import pandas as pd

        frame = pd.read_csv(csv_path)
        if not {"text", "label"}.issubset(frame.columns):
            raise ValueError(f"{csv_path} must contain 'text' and 'label' columns.")
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive when provided.")
            if max_samples > len(frame):
                raise ValueError(
                    f"Requested {max_samples} samples from {csv_path}, which has only {len(frame)} rows."
                )
            frame = frame.sample(n=max_samples, random_state=random_state)
        self.samples = list(zip(frame["text"].astype(str), frame["label"].astype(int)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        return self.samples[index]


def _classification_collate(tokenizer: object, max_length: int):
    def collate(batch: Iterable[Tuple[str, int]]) -> Dict[str, torch.Tensor]:
        texts, labels = zip(*batch)
        encoded = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids1": encoded["input_ids"],
            "attention_mask1": encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def extract_pooler_embeddings(
    model: torch.nn.Module,
    data_loader: DataLoader[Dict[str, torch.Tensor]],
    dimension: int,
) -> Tuple[List[object], List[int]]:
    """Extract the nested-prefix evaluation embeddings.

    Reads z through ``models.tinybert.mrl_sentence_embedding`` -- the exact
    helper the MRL loss uses -- so the probe can never truncate a different
    vector than the one whose prefixes were supervised. The legacy function name
    is retained for API compatibility.
    """
    device = model.device
    embeddings: List[object] = []
    labels: List[int] = []
    with torch.no_grad():
        for batch in data_loader:
            model_inputs = {
                "input_ids": batch["input_ids1"].to(device),
                "attention_mask": batch["attention_mask1"].to(device),
            }
            try:
                output = model(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            except TypeError as exc:
                if "output_hidden_states" not in str(exc) and "return_dict" not in str(exc):
                    raise
                output = model(**model_inputs)

            embedding = mrl_sentence_embedding(output)
            embeddings.extend(embedding[:, :dimension].cpu().numpy())
            labels.extend(batch["labels"].tolist())
    return embeddings, labels


def validation_probe_score(
    model: torch.nn.Module,
    tokenizer: object,
    train_csv: str | Path,
    config: "MRLEvaluationConfig",
    holdout: float = 0.2,
    max_rows: Optional[int] = 6000,
) -> float:
    """Mean macro-F1 of a probe fitted and scored INSIDE the probe-train split.

    Used to choose which epoch's encoder to keep. It never touches the test set:
    the probe-train rows are split into a fit part and a held-out part, so epoch
    selection cannot leak test labels. Returns the mean across all nested
    dimensions, which is a steadier selection signal than any single width.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    import numpy as np

    was_training = model.training
    model.eval()
    collate = _classification_collate(tokenizer, config.max_length)
    # Epoch selection only needs a relative signal, so it subsamples for speed.
    # The reported evaluation always uses the full split.
    dataset = ClassificationDataset(
        train_csv, max_samples=max_rows, random_state=config.random_state
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    embeddings, labels = extract_pooler_embeddings(model, loader, max(config.dimensions))
    matrix, targets = np.array(embeddings), np.array(labels)

    rng = np.random.RandomState(config.random_state)
    order = rng.permutation(len(targets))
    cut = int(len(order) * (1.0 - holdout))
    fit_idx, held_idx = order[:cut], order[cut:]

    scores = []
    for dimension in config.dimensions:
        classifier = LogisticRegression(random_state=config.random_state, max_iter=config.max_iter)
        classifier.fit(matrix[fit_idx][:, :dimension], targets[fit_idx])
        predictions = classifier.predict(matrix[held_idx][:, :dimension])
        scores.append(f1_score(targets[held_idx], predictions, average="macro"))

    if was_training:
        model.train()
    return float(np.mean(scores))


def evaluate_tweeteval(
    model: torch.nn.Module,
    tokenizer: object,
    train_csv: str | Path,
    test_csv: str | Path,
    config: MRLEvaluationConfig = MRLEvaluationConfig(),
) -> Dict[int, Dict[str, float]]:
    """Run the baseline TweetEval linear-probe evaluation at each MRL dimension."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    collate = _classification_collate(tokenizer, config.max_length)
    train_loader = DataLoader(
        ClassificationDataset(
            train_csv,
            max_samples=config.train_samples,
            random_state=config.random_state,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        ClassificationDataset(
            test_csv,
            max_samples=config.test_samples,
            random_state=config.random_state,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    import numpy as np

    max_dim = max(config.dimensions)
    all_train_embeddings, train_labels = extract_pooler_embeddings(model, train_loader, max_dim)
    all_test_embeddings, test_labels = extract_pooler_embeddings(model, test_loader, max_dim)
    train_mat = np.array(all_train_embeddings)
    test_mat = np.array(all_test_embeddings)

    results: Dict[int, Dict[str, float]] = {}
    for dimension in config.dimensions:
        # The encoder has already produced full-width embeddings above. Reuse
        # their leading dimensions instead of running TinyBERT again for each
        # nested Matryoshka size. This preserves the evaluation exactly while
        # avoiding 6 redundant train/test encoder passes.
        train_embeddings = train_mat[:, :dimension]
        test_embeddings = test_mat[:, :dimension]
        classifier = LogisticRegression(
            random_state=42, max_iter=config.max_iter, verbose=0
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(train_embeddings, train_labels)
            converged = not any(
                issubclass(w.category, ConvergenceWarning) for w in caught
            )
        predictions = classifier.predict(test_embeddings)
        results[dimension] = {
            "accuracy": float(accuracy_score(test_labels, predictions)),
            "f1": float(f1_score(test_labels, predictions, average="macro")),
            "probe_n_iter": int(classifier.n_iter_[0]),
            "probe_converged": bool(converged),
        }
    return results
