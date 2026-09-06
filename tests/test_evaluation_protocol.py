"""Regression tests for the TweetEval evaluation protocol."""

from pathlib import Path

from evaluate import ClassificationDataset, MRLEvaluationConfig


def test_tweeteval_eval_can_use_a_deterministic_3000_sample() -> None:
    train_csv = Path("data/tweet_train.csv")
    ds1 = ClassificationDataset(train_csv, max_samples=3000, random_state=42)
    ds2 = ClassificationDataset(train_csv, max_samples=3000, random_state=42)
    assert len(ds1) == len(ds2) == 3000
    assert ds1[0] == ds2[0]
    assert all(isinstance(label, int) for _, label in ds1.samples[:10])


def test_default_evaluation_config_preserves_full_dataset_behavior() -> None:
    config = MRLEvaluationConfig()
    assert config.train_samples is None
    assert config.max_length == 256


def test_paper_runner_config_is_representable() -> None:
    config = MRLEvaluationConfig(train_samples=3000, random_state=42)
    assert config.train_samples == 3000
    assert config.random_state == 42


def test_probe_trains_on_the_full_tweeteval_split_by_default() -> None:
    """Table 6's "Train (Sampled) 3,000" is the CONTRASTIVE corpus, not the probe.

    A.2 states the 3,000 TweetEval rows are sampled into the training corpus
    ("6,000 sentences from classification datasets, 3,000 per dataset"). A.4
    describes the probe as the standard Conneau & Kiela logistic regression on
    frozen features, which uses the whole train split -- as do the official
    notebook and the reference MRL baseline.

    Measured on one frozen untrained TinyBERT-6L: a 3,000-row probe scores
    51.11 macro-F1 at d=16 and 69.23 at d=768; the full 26,732-row probe scores
    62.62 and 71.99 on the identical encoder.
    """
    from pathlib import Path as _Path

    import pandas as pd

    config = MRLEvaluationConfig()
    assert config.train_samples is None, "the probe must use the whole split"

    root = _Path(__file__).resolve().parents[1]
    source = (root / "run_experiments.py").read_text(encoding="utf-8")
    assert '_resolve_data_file("tweet_train.csv", "MIPIC_EVAL_TRAIN")' in source
    assert '_resolve_data_file("tweet_train_3k.csv"' not in source

    full = pd.read_csv(root / "data" / "tweet_train.csv")
    assert len(full) > 20000, f"expected the full split, got {len(full)} rows"
    # And it must stay disjoint from the test split.
    test = pd.read_csv(root / "data" / "tweet_test.csv")
    assert not set(full["text"].astype(str)) & set(test["text"].astype(str))


def test_epoch_selection_probe_subsamples_for_speed_only() -> None:
    """Selection may subsample; the reported evaluation may not."""
    from pathlib import Path as _Path
    import inspect

    from evaluate import validation_probe_score

    signature = inspect.signature(validation_probe_score)
    assert signature.parameters["max_rows"].default == 6000

    source = (_Path(__file__).resolve().parents[1] / "evaluate.py").read_text(encoding="utf-8")
    start = source.index("def evaluate_tweeteval")
    assert "max_samples=config.train_samples" in source[start:], (
        "the reported evaluation must honour train_samples (None = full split)"
    )


def test_the_classification_benchmark_is_tweet_sentiment_extraction() -> None:
    """Record what the "TweetEval" files actually are, so it cannot drift.

    They are Kaggle *Tweet Sentiment Extraction*, not canonical TweetEval
    (Barbieri et al. 2020), whose sentiment split has 45,615 / 2,000 / 12,284
    rows. Ours are 26,732 / 3,432 with 10-character hex textIDs.

    The paper uses the same non-canonical file: Table 6 lists TweetEval test =
    3,434, two rows off this one and nothing like 12,284. So comparisons against
    the paper's TweetEval row are on matching data -- but the *name* is wrong
    upstream, and anything written up should say so.
    """
    from pathlib import Path as _Path

    import pandas as pd

    root = _Path(__file__).resolve().parents[1]
    test = pd.read_csv(root / "data" / "tweet_test.csv")
    train = pd.read_csv(root / "data" / "tweet_train.csv")

    assert len(test) == 3432 and len(train) == 26732
    assert list(test.columns) == ["id", "text", "label", "label_text"]
    # Tweet Sentiment Extraction textID format.
    assert set(test["id"].astype(str).str.len()) == {10}
    assert set(test["label_text"]) == {"negative", "neutral", "positive"}
    assert sorted(test["label"].unique().tolist()) == [0, 1, 2]

    # Emphatically NOT the canonical TweetEval sentiment test split.
    assert len(test) != 12284

    # Splits must stay disjoint.
    assert not set(train["text"].astype(str)) & set(test["text"].astype(str))
