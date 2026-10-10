import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# CONFIG
# ============================================================

DIMENSIONS = [
    16, 32, 64, 128, 256,
    512, 768, 1024, 2048, 4096
]

TRAIN_EMB = "results/flowvec_mrl/english_train_embeddings.npy"
TEST_EMB = "results/flowvec_mrl/english_test_embeddings.npy"

TRAIN_DATA = "data/tweet_train.csv"
TEST_DATA = "data/tweet_test.csv"

RESULTS_PATH = "results/flowvec_mrl/english_mrl_results.csv"

RANDOM_STATE = 42
MAX_ITER = 1000


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("Loading data")
print("=" * 70)

X_train_full = np.load(TRAIN_EMB)
X_test_full = np.load(TEST_EMB)

train_df = pd.read_csv(TRAIN_DATA)
test_df = pd.read_csv(TEST_DATA)

y_train = train_df["label"].to_numpy()
y_test = test_df["label"].to_numpy()

print(f"Train embeddings: {X_train_full.shape}")
print(f"Test embeddings : {X_test_full.shape}")
print(f"Train labels    : {y_train.shape}")
print(f"Test labels     : {y_test.shape}")

assert X_train_full.shape[0] == len(y_train)
assert X_test_full.shape[0] == len(y_test)


# ============================================================
# EVALUATION
# ============================================================

results = []

print("\n" + "=" * 70)
print("DOWNSTREAM MRL EVALUATION")
print("=" * 70)

for dim in DIMENSIONS:

    print(f"\n{'-' * 70}")
    print(f"Dimension: {dim}")
    print(f"{'-' * 70}")

    # Nested representation
    X_train = X_train_full[:, :dim]
    X_test = X_test_full[:, :dim]

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")

    # Logistic Regression
    clf = LogisticRegression(
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )

    clf.fit(X_train, y_train)

    # Prediction
    y_pred = clf.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    print(f"Accuracy : {accuracy:.6f}")
    print(f"Macro-F1 : {macro_f1:.6f}")

    results.append({
        "dimension": dim,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    })


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))

os.makedirs(
    os.path.dirname(RESULTS_PATH),
    exist_ok=True
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print(f"\nSaved results to:")
print(RESULTS_PATH)