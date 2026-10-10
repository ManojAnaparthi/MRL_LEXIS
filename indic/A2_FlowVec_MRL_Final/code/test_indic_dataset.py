import argparse
import numpy as np
from datasets import get_dataset_config_names, load_dataset

from common import resolve_label_column, resolve_text_column

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mteb/IndicSentiment")
parser.add_argument("--language", default="hi")
parser.add_argument("--num-samples", type=int, default=4)
args = parser.parse_args()

print("=== MTEB IndicSentiment dataset test ===")
print("Dataset:", args.dataset)
print("Requested language/config:", args.language)

configs = get_dataset_config_names(args.dataset)
print("\nAvailable configs:")
print(configs)

if args.language not in configs:
    raise SystemExit(
        f"\nConfig '{args.language}' was not found.\n"
        "Do not guess the config name; inspect the list above."
    )

print(f"\nLoading config '{args.language}' ...")
dataset = load_dataset(args.dataset, args.language)

print("Splits:", list(dataset.keys()))

if "train" not in dataset or "test" not in dataset:
    raise SystemExit(
        f"Expected train/test splits, got {list(dataset.keys())}"
    )

train = dataset["train"]
test = dataset["test"]

print("\nTrain columns:", train.column_names)
print("Test columns:", test.column_names)
print("Train rows:", len(train))
print("Test rows:", len(test))

text_col = resolve_text_column(train.column_names)
label_col = resolve_label_column(train.column_names)

print("Resolved text column:", text_col)
print("Resolved label column:", label_col)

labels = np.asarray(train[label_col])
print("Unique train labels:", np.unique(labels))
print("Number of classes:", len(np.unique(labels)))

print("\nSample rows:")
for row in train.select(range(min(args.num_samples, len(train)))):
    print({
        "text": row[text_col],
        "label": row[label_col],
    })

if len(np.unique(labels)) < 2:
    raise SystemExit("Dataset does not contain at least two classes.")

print("\nINDIC DATASET TEST PASSED")
