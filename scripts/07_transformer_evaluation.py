#!/usr/bin/env python
"""
Evaluate a fine-tuned transformer on the held-out test split.

Run this once, after training and after reviewing validation metrics. Repeated
test evaluation would invalidate the comparison with the classical baseline.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from data.loader import load_preprocessed_data
from evaluation.visualization import EvaluationVisualizer
from models.transformer import (
    DEFAULT_MAX_LENGTH,
    SEVERITY_LABELS,
    TEXT_COLUMN,
    classification_metrics,
    ids_to_labels,
    labels_to_ids,
    load_split_indices,
    predict_proba,
)
from utils.config import (
    BEST_MODEL_PATH,
    FIGURES_DIR,
    MODEL_DIR,
    PREPROCESSED_DATA_PATH,
    RESULTS_DIR,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASELINE_TEST_MACRO_F1 = 0.3402
TARGET_GAIN = 0.03


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR / "transformer-severity",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Defaults to the value stored in training_metadata.json.",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=BEST_MODEL_PATH,
        help="Classical model used for the same-test comparison.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Do not evaluate the classical model on the test split.",
    )
    parser.add_argument(
        "--no-fp16", action="store_true", help="Disable fp16 inference on CUDA."
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing evaluation. This invalidates the one-shot test rule.",
    )
    return parser.parse_args()


def evaluate_baseline(model_path: Path, test_frame: pd.DataFrame):
    """Evaluate the classical model on exactly the same test rows."""
    from models.train import ModelTrainer

    bundle = ModelTrainer.load_model(model_path)
    model = bundle["model"]
    encoder = bundle["label_encoder"]
    predictions = encoder.inverse_transform(model.predict(test_frame[[TEXT_COLUMN]]))
    return predictions


def format_metrics(metrics):
    return (
        f"macro F1 {metrics['f1_macro']:.4f} | "
        f"weighted F1 {metrics['f1_weighted']:.4f} | "
        f"accuracy {metrics['accuracy']:.4f}"
    )


def main():
    args = parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(args.model_dir)
    metadata_path = model_dir / "training_metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"Missing training metadata: {metadata_path}")

    evaluation_path = args.results_dir / "transformer_evaluation.json"
    if evaluation_path.exists() and not args.force:
        raise SystemExit(
            f"Test evaluation already exists: {evaluation_path}\n"
            "The test split is one-shot. Pass --force only if the previous "
            "evaluation was invalid."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    max_length = args.max_length or metadata.get("max_length", DEFAULT_MAX_LENGTH)
    split_path = model_dir / "split_indices.json"
    splits = load_split_indices(split_path)

    data = load_preprocessed_data(PREPROCESSED_DATA_PATH)
    test_positions = splits["test"]
    test_frame = data.iloc[test_positions].reset_index(drop=True)
    test_texts = test_frame[TEXT_COLUMN].astype(str).tolist()
    y_true_labels = test_frame["severity_category"].astype(str).to_numpy()
    y_true_ids = labels_to_ids(y_true_labels)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning("CUDA is unavailable; evaluating on CPU.")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    probabilities = predict_proba(
        model,
        tokenizer,
        test_texts,
        batch_size=args.batch_size,
        max_length=max_length,
        device=device,
        fp16=(device == "cuda" and not args.no_fp16),
    )
    y_pred_ids = np.argmax(probabilities, axis=-1)
    y_pred_labels = ids_to_labels(y_pred_ids)
    confidence = probabilities.max(axis=1)

    transformer_metrics = classification_metrics(y_true_ids, y_pred_ids)
    transformer_report = classification_report(
        y_true_labels,
        y_pred_labels,
        labels=SEVERITY_LABELS,
        digits=4,
        zero_division=0,
    )

    baseline_metrics = None
    baseline_predictions = None
    if not args.skip_baseline and Path(args.baseline_model).exists():
        logger.info("Evaluating classical baseline on the same test rows")
        baseline_predictions = evaluate_baseline(Path(args.baseline_model), test_frame)
        baseline_metrics = classification_metrics(
            y_true_ids, labels_to_ids(baseline_predictions)
        )
        if abs(baseline_metrics["f1_macro"] - BASELINE_TEST_MACRO_F1) > 0.002:
            logger.warning(
                "Baseline macro F1 is %.4f, expected ~%.4f. "
                "The test split may differ from the released baseline.",
                baseline_metrics["f1_macro"],
                BASELINE_TEST_MACRO_F1,
            )
    elif not args.skip_baseline:
        logger.warning("Baseline model not found: %s", args.baseline_model)

    target_f1 = BASELINE_TEST_MACRO_F1 + TARGET_GAIN
    criteria = {
        "macro_f1_at_least_target": bool(transformer_metrics["f1_macro"] >= target_f1)
    }
    if baseline_metrics is not None:
        criteria["blocker_recall_not_reduced"] = bool(
            transformer_metrics["blocker_recall"] >= baseline_metrics["blocker_recall"]
        )
        criteria["critical_recall_not_reduced"] = bool(
            transformer_metrics["critical_recall"]
            >= baseline_metrics["critical_recall"]
        )
    adopt = all(criteria.values())

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    prediction_frame = test_frame[
        [
            column
            for column in (
                "bug_id",
                "short_description",
                "long_description",
                "severity_category",
            )
            if column in test_frame.columns
        ]
    ].copy()
    prediction_frame["predicted_severity"] = y_pred_labels
    prediction_frame["confidence"] = confidence
    prediction_frame["correct"] = y_true_labels == y_pred_labels
    for index, label in enumerate(SEVERITY_LABELS):
        prediction_frame[f"prob_{label}"] = probabilities[:, index]
    prediction_frame.to_csv(
        args.results_dir / "transformer_test_predictions.csv", index=False
    )
    prediction_frame.loc[~prediction_frame["correct"]].to_csv(
        args.results_dir / "transformer_misclassified_cases.csv", index=False
    )

    confusion = pd.crosstab(
        pd.Categorical(y_true_labels, categories=SEVERITY_LABELS),
        pd.Categorical(y_pred_labels, categories=SEVERITY_LABELS),
        dropna=False,
    )
    confusion.index.name = "true_label"
    confusion.columns.name = "predicted_label"
    confusion.to_csv(args.results_dir / "transformer_confusion_matrix.csv")

    visualizer = EvaluationVisualizer(SEVERITY_LABELS)
    visualizer.plot_confusion_matrix(
        y_true_labels,
        y_pred_labels,
        title="Transformer Confusion Matrix (Test Set)",
        save_path=args.figures_dir / "transformer_confusion_matrix.png",
    )

    evaluation = {
        "model_dir": str(model_dir),
        "model_name": metadata.get("model_name"),
        "test_rows": int(len(test_positions)),
        "max_length": max_length,
        "device": device,
        "transformer": transformer_metrics,
        "baseline": baseline_metrics,
        "baseline_reference_macro_f1": BASELINE_TEST_MACRO_F1,
        "target_macro_f1": target_f1,
        "adoption_criteria": criteria,
        "adopt": adopt,
    }
    (evaluation_path).write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    lines = [
        "TRANSFORMER TEST EVALUATION",
        "=" * 70,
        "",
        f"Model: {metadata.get('model_name')}",
        f"Test rows: {len(test_positions)}",
        f"Max length: {max_length}",
        f"Device: {device}",
        "",
        "TRANSFORMER",
        "-" * 70,
        format_metrics(transformer_metrics),
        transformer_report,
        "",
        "CLASSICAL BASELINE (same test rows)",
        "-" * 70,
    ]
    if baseline_metrics is None:
        lines.append("Not evaluated.")
    else:
        delta_macro_f1 = transformer_metrics["f1_macro"] - baseline_metrics["f1_macro"]
        lines.extend(
            [
                format_metrics(baseline_metrics),
                f"Delta macro F1: {delta_macro_f1:+.4f}",
                "",
                f"Blocker recall: baseline {baseline_metrics['blocker_recall']:.4f}"
                f" -> transformer {transformer_metrics['blocker_recall']:.4f}",
                f"Critical recall: baseline {baseline_metrics['critical_recall']:.4f}"
                f" -> transformer {transformer_metrics['critical_recall']:.4f}",
                "",
                "PER-CLASS F1 COMPARISON",
                "-" * 70,
                f"{'class':10s} {'baseline':>10s} {'transformer':>12s} "
                f"{'delta':>9s}",
            ]
        )
        for label in SEVERITY_LABELS:
            baseline_f1 = baseline_metrics[f"{label}_f1"]
            transformer_f1 = transformer_metrics[f"{label}_f1"]
            lines.append(
                f"{label:10s} {baseline_f1:10.4f} {transformer_f1:12.4f} "
                f"{transformer_f1 - baseline_f1:+9.4f}"
            )

    lines.extend(
        [
            "",
            "ADOPTION RULE",
            "-" * 70,
            f"Required macro F1: >= {target_f1:.4f}",
        ]
    )
    for name, passed in criteria.items():
        lines.append(f"{name}: {'PASS' if passed else 'FAIL'}")
    lines.append(f"Adopt transformer: {'YES' if adopt else 'NO'}")

    report_text = "\n".join(lines)
    (args.results_dir / "transformer_evaluation_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    print(report_text)


if __name__ == "__main__":
    main()
