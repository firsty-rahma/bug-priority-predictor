"""
Transformer helpers for bug severity classification.

The transformer is an experimental alternative to the classical TF-IDF model.
It keeps the same six-class labels and the same 80/20 train/test split so the
test result is directly comparable with the released baseline.
"""

import contextlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer

TEXT_COLUMN = "text_for_model"

# Same order as utils.config.SEVERITY_CLASSES and the classical LabelEncoder.
SEVERITY_LABELS = ["blocker", "critical", "major", "minor", "normal", "trivial"]
LABEL_TO_ID = {label: index for index, label in enumerate(SEVERITY_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
NUM_LABELS = len(SEVERITY_LABELS)

DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 256


def labels_to_ids(labels: Iterable[str]) -> np.ndarray:
    """Map severity names to contiguous integer ids."""
    return np.asarray([LABEL_TO_ID[label] for label in labels], dtype=np.int64)


def ids_to_labels(ids: Iterable[int]) -> np.ndarray:
    """Map integer ids back to severity names."""
    return np.asarray([ID_TO_LABEL[int(index)] for index in ids], dtype=object)


def stratified_split_indices(
    labels: Sequence[str],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted train/validation/test row positions.

    ``test_size`` matches the classical split. ``val_size`` is taken from the
    remaining training rows and is used only for early stopping.
    """
    labels_array = np.asarray(labels)
    positions = np.arange(len(labels_array))

    train_val_positions, test_positions = train_test_split(
        positions,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_array,
    )
    train_positions, val_positions = train_test_split(
        train_val_positions,
        test_size=val_size,
        random_state=random_state,
        stratify=labels_array[train_val_positions],
    )

    return (
        np.sort(train_positions),
        np.sort(val_positions),
        np.sort(test_positions),
    )


def save_split_indices(
    path: Path,
    train_positions: Sequence[int],
    val_positions: Sequence[int],
    test_positions: Sequence[int],
) -> None:
    """Persist split positions so evaluation reproduces the training split."""
    payload = {
        "train": [int(index) for index in train_positions],
        "validation": [int(index) for index in val_positions],
        "test": [int(index) for index in test_positions],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_indices(path: Path) -> Dict[str, np.ndarray]:
    """Load persisted split positions."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "train": np.asarray(payload["train"], dtype=np.int64),
        "validation": np.asarray(payload["validation"], dtype=np.int64),
        "test": np.asarray(payload["test"], dtype=np.int64),
    }


def compute_class_weights(
    labels: Sequence[int], num_classes: int = NUM_LABELS
) -> np.ndarray:
    """Balanced class weights: ``n / (num_classes * count)``."""
    labels_array = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_array, minlength=num_classes)
    if len(counts) != num_classes:
        raise ValueError(f"Expected {num_classes} classes, found {len(counts)}")
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Cannot weight missing classes: {missing}")

    return len(labels_array) / (num_classes * counts)


class BugTextDataset(Dataset):
    """Tokenized bug reports with integer severity labels."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: Any,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
        )
        self.labels = [int(label) for label in labels]

        if len(self.encodings["input_ids"]) != len(self.labels):
            raise ValueError("Tokenized texts and labels have different lengths")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(values[index]) for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy for rare severities."""

    def __init__(
        self,
        *args,
        class_weights: Optional[Sequence[float]] = None,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
        self.label_smoothing = float(label_smoothing)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            weight=weight,
            label_smoothing=self.label_smoothing,
        )

        if return_outputs:
            return loss, outputs
        return loss


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[str, float]:
    """Macro/weighted metrics plus per-class precision, recall, and F1."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    labels = list(range(NUM_LABELS))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_array,
        y_pred_array,
        labels=labels,
        zero_division=0,
    )

    metrics = {
        "f1_macro": float(f1_score(y_true_array, y_pred_array, average="macro")),
        "f1_weighted": float(f1_score(y_true_array, y_pred_array, average="weighted")),
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
    }

    for index, label in enumerate(SEVERITY_LABELS):
        metrics[f"{label}_precision"] = float(precision[index])
        metrics[f"{label}_recall"] = float(recall[index])
        metrics[f"{label}_f1"] = float(f1[index])

    return metrics


def compute_metrics(eval_prediction) -> Dict[str, float]:
    """Hugging Face Trainer metric callback."""
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return classification_metrics(labels, predictions)


def predict_proba(
    model: nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    batch_size: int = 16,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: Optional[str] = None,
    fp16: Optional[bool] = None,
) -> np.ndarray:
    """Return class probabilities while keeping inference in eval mode."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if fp16 is None:
        fp16 = device == "cuda"

    model.to(device)
    model.eval()
    probabilities: List[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if fp16 and device == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            logits = model(**encoded).logits

        batch_probabilities = torch.softmax(logits.float(), dim=-1)
        probabilities.append(batch_probabilities.cpu().numpy())

    if not probabilities:
        return np.empty((0, NUM_LABELS), dtype=np.float32)
    return np.concatenate(probabilities, axis=0)
