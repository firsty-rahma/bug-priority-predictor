"""
Tests for transformer data, splitting, weighting, and inference helpers.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from torch import nn

from models.transformer import (
    LABEL_TO_ID,
    NUM_LABELS,
    SEVERITY_LABELS,
    BugTextDataset,
    classification_metrics,
    compute_class_weights,
    ids_to_labels,
    labels_to_ids,
    predict_proba,
    stratified_split_indices,
)


class FakeTokenizer:
    """Minimal tokenizer that avoids a model download in unit tests."""

    def __call__(
        self,
        texts,
        truncation=True,
        max_length=256,
        padding=False,
        return_tensors=None,
    ):
        if isinstance(texts, str):
            texts = [texts]
        input_ids = [
            list(range(1, min(len(text.split()), max_length) + 1)) for text in texts
        ]
        if not input_ids:
            input_ids = [[]]

        if padding:
            width = max((len(row) for row in input_ids), default=1)
            attention = [[1] * len(row) + [0] * (width - len(row)) for row in input_ids]
            input_ids = [row + [0] * (width - len(row)) for row in input_ids]
        else:
            attention = [[1] * len(row) for row in input_ids]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attention}


class FakeModel(nn.Module):
    """Returns fixed logits based on the number of input rows."""

    def __init__(self, num_rows):
        super().__init__()
        self.num_rows = num_rows

    def forward(self, input_ids, attention_mask=None):
        logits = torch.zeros((input_ids.shape[0], NUM_LABELS), dtype=torch.float32)
        logits[:, -1] = 2.0
        return SimpleNamespace(logits=logits)


class TestLabels:
    def test_round_trip(self):
        ids = labels_to_ids(SEVERITY_LABELS)
        assert ids.tolist() == list(range(NUM_LABELS))
        assert ids_to_labels(ids).tolist() == SEVERITY_LABELS

    def test_all_classes_mapped(self):
        assert set(LABEL_TO_ID) == set(SEVERITY_LABELS)

    def test_unknown_label_raises(self):
        with pytest.raises(KeyError):
            labels_to_ids(["unknown"])


class TestSplits:
    def test_splits_are_disjoint_and_stratified(self):
        labels = []
        for label in SEVERITY_LABELS:
            labels.extend([label] * 20)

        train, validation, test = stratified_split_indices(
            labels, test_size=0.2, val_size=0.1, random_state=42
        )

        assert len(train) + len(validation) + len(test) == len(labels)
        assert not set(train) & set(validation)
        assert not set(train) & set(test)
        assert not set(validation) & set(test)
        for split in (train, validation, test):
            assert set(pd.Series(np.asarray(labels)[split])) == set(SEVERITY_LABELS)


class TestClassWeights:
    def test_rare_classes_get_larger_weights(self):
        labels = np.array([0] * 40 + [1] * 20 + [2] * 10 + [3] * 5 + [4] * 4 + [5] * 1)
        weights = compute_class_weights(labels)

        assert len(weights) == NUM_LABELS
        assert (
            weights[5] > weights[4] > weights[3] > weights[2] > weights[1] > weights[0]
        )
        assert np.isclose(weights[0], len(labels) / (NUM_LABELS * 40))

    def test_missing_class_raises(self):
        with pytest.raises(ValueError):
            compute_class_weights([0, 0, 1, 1])


class TestDataset:
    def test_tokenizes_and_returns_labels(self):
        dataset = BugTextDataset(
            ["crash on startup", "add dark mode"],
            [0, 4],
            FakeTokenizer(),
            max_length=8,
        )

        item = dataset[0]
        assert len(dataset) == 2
        assert item["labels"].item() == 0
        assert item["input_ids"].dtype == torch.long
        assert item["attention_mask"].sum().item() == 3


class TestMetrics:
    def test_perfect_predictions(self):
        labels = list(range(NUM_LABELS))
        metrics = classification_metrics(labels, labels)

        assert metrics["f1_macro"] == pytest.approx(1.0)
        assert metrics["f1_weighted"] == pytest.approx(1.0)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["blocker_recall"] == pytest.approx(1.0)


class TestInference:
    def test_probabilities_sum_to_one(self):
        model = FakeModel(num_rows=3)
        probabilities = predict_proba(
            model,
            FakeTokenizer(),
            ["one", "two words", "three word report"],
            batch_size=2,
            max_length=8,
            device="cpu",
            fp16=False,
        )

        assert probabilities.shape == (3, NUM_LABELS)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert probabilities.argmax(axis=1).tolist() == [5, 5, 5]
