from pathlib import Path

import pandas as pd

from prepare_training_corpus import PUBLISHED_TRAINING_SOURCES, prepare_training_corpus


def test_bundled_paper_training_corpus_matches_supplied_pool() -> None:
    corpus = pd.read_csv(Path("data/paper_train.csv"))
    supplied = pd.read_csv(Path("data/final_data.csv"))[["text", "source"]]
    assert list(corpus.columns) == ["text", "source"]
    pd.testing.assert_frame_equal(corpus, supplied)
    assert len(corpus) == 20240
    assert "DGEM_FORMAT" in set(corpus["source"])


def test_training_corpus_preparation_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "paper_train.csv"
    prepared = prepare_training_corpus("data/final_data.csv", output)
    saved = pd.read_csv(output)
    pd.testing.assert_frame_equal(prepared, saved)
    assert len(saved) == 20240
    assert set(saved["source"].drop_duplicates().tolist()) == set(PUBLISHED_TRAINING_SOURCES)
