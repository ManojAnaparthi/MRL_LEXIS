from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# Configuration
# ============================================================

LANGUAGES = ["hi", "bn", "gu", "ta", "te"]

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

EMBED_DIR = Path("results/flowvec_mrl_indic/indic")

DATA_DIR = Path("data/indic")

OUTPUT_FILE = Path("results/flowvec_mrl_indic/indic_mrl_results.csv")


# ============================================================
# Evaluation
# ============================================================

results = []


for lang in LANGUAGES:

    print("\n" + "=" * 60)
    print(f"Language: {lang}")
    print("=" * 60)

    # Load embeddings
    X_train = np.load(
        EMBED_DIR / f"{lang}_train_mrl.npy"
    )

    X_test = np.load(
        EMBED_DIR / f"{lang}_test_mrl.npy"
    )

    # Load labels
    train_df = pd.read_csv(
        DATA_DIR / f"{lang}_train.csv"
    )

    test_df = pd.read_csv(
        DATA_DIR / f"{lang}_test.csv"
    )

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    print(
        "Train:",
        X_train.shape,
        "Test:",
        X_test.shape,
    )

    # --------------------------------------------------------
    # Evaluate every nested dimension
    # --------------------------------------------------------

    for dim in DIMENSIONS:

        X_train_dim = X_train[:, :dim]
        X_test_dim = X_test[:, :dim]

        clf = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )

        clf.fit(
            X_train_dim,
            y_train,
        )

        predictions = clf.predict(
            X_test_dim
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
        )

        results.append(
            {
                "language": lang,
                "dimension": dim,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )

        print(
            f"{dim:4d}D | "
            f"Accuracy: {accuracy:.6f} | "
            f"Macro-F1: {macro_f1:.6f}"
        )


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(results)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 60)
print("Results saved to:")
print(OUTPUT_FILE)
print("=" * 60)


# ============================================================
# Average across the five languages
# ============================================================

mean_results = (
    results_df
    .groupby("dimension")
    [["accuracy", "macro_f1"]]
    .mean()
    .reset_index()
)

print("\nMean across 5 languages:")
print(mean_results.to_string(index=False))