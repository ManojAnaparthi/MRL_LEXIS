from pathlib import Path

import torch

from evaluate import MRLEvaluationConfig, evaluate_tweeteval


class _FakeTokenizer:
    def __call__(self, texts, **kwargs):
        n = len(texts)
        return {
            "input_ids": torch.ones((n, 4), dtype=torch.long),
            "attention_mask": torch.ones((n, 4), dtype=torch.long),
        }


class _FakeOutput:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class _CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._device = torch.device("cpu")
        self.forward_calls = 0

    @property
    def device(self):
        return self._device

    def forward(self, input_ids, attention_mask):
        self.forward_calls += 1
        # Batch index is encoded into feature rows so the classifier still sees
        # a deterministic, separable toy problem.
        base = input_ids[:, :1].float()
        features = torch.cat([base, base * 2, base * 3, base * 4], dim=1)
        return _FakeOutput(features)


def test_tweeteval_reuses_full_embeddings_for_all_dimensions(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    train_csv.write_text(
        "text,label\n"
        "a,0\n"
        "b,0\n"
        "c,1\n"
        "d,1\n",
        encoding="utf-8",
    )
    test_csv.write_text(
        "text,label\n"
        "e,0\n"
        "f,1\n",
        encoding="utf-8",
    )

    model = _CountingModel()
    config = MRLEvaluationConfig(dimensions=(2, 4), batch_size=2, max_length=4)

    evaluate_tweeteval(model, _FakeTokenizer(), train_csv, test_csv, config)

    # One encoder pass per batch for train and test, regardless of the number
    # of nested dimensions requested.
    assert model.forward_calls == 3
