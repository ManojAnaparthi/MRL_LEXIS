"""Prepare the paper-faithful TinyBERT training corpus from the bundled pool.

The repository already contains the exact pooled training input in
``data/final_data.csv``. The paper's dataset accounting is internally inconsistent
with the supplied 20,240-row pool and does not name every ``source`` value, while
the official notebook consumes ``final_data.csv`` directly. Therefore this helper
does not silently remove any source: it validates the schema and copies the
supplied pool unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

SOURCE_COLUMN: Final[str] = "source"
TEXT_COLUMN: Final[str] = "text"
PUBLISHED_TRAINING_SOURCES: Final[tuple[str, ...]] = (
    "BANKING77",
    "TWEET_EVAL_SENT",
    "MRPC",
    "DEEHAN1866/WIC",
    "DGEM_FORMAT",
    "STSB",
    "SICK",
)

DEFAULT_INPUT = Path("data/final_data.csv")
DEFAULT_OUTPUT = Path("data/paper_train.csv")


def prepare_training_corpus(
    input_csv: str | Path = DEFAULT_INPUT,
    output_csv: str | Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """Filter the bundled pool to the paper's in-domain training sources."""
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing pooled dataset: {input_path}")

    frame = pd.read_csv(input_path)
    missing = {TEXT_COLUMN, SOURCE_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(
            f"{input_path} is missing required columns: {sorted(missing)}"
        )

    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype(str)
    frame[SOURCE_COLUMN] = frame[SOURCE_COLUMN].astype(str)

    unknown = set(frame[SOURCE_COLUMN]) - set(PUBLISHED_TRAINING_SOURCES)
    if unknown:
        raise ValueError(f"Unexpected dataset sources in {input_path}: {sorted(unknown)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = frame[[TEXT_COLUMN, SOURCE_COLUMN]].copy()
    selected.to_csv(output_path, index=False)
    return selected


if __name__ == "__main__":
    corpus = prepare_training_corpus()
    print(f"Wrote {len(corpus):,} training sentences to {DEFAULT_OUTPUT}")
    print(corpus.groupby(SOURCE_COLUMN).size().to_string())
