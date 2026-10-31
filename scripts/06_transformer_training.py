#!/usr/bin/env python
"""
Fine-tune a transformer for six-class bug severity classification.

The test split is never evaluated here. It is reserved for
``07_transformer_evaluation.py`` and should be touched once.
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.loader import load_preprocessed_data
from models.transformer import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    LABEL_TO_ID,
    NUM_LABELS,
    SEVERITY_LABELS,
    TEXT_COLUMN,
    BugTextDataset,
    WeightedTrainer,
    compute_class_weights,
    compute_metrics,
    labels_to_ids,
    save_split_indices,
    stratified_split_indices,
)
from utils.config import (
    MODEL_DIR,
    PREPROCESSED_DATA_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--output-dir", type=Path, default=MODEL_DIR / "transformer-severity"
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Stratified training subset size for smoke tests (0 = all).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=0,
        help="Stratified validation subset size for smoke tests (0 = all).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override epochs for smoke tests (0 or -1 = use epochs).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Epochs without validation improvement before stopping.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade speed for lower GPU memory.",
    )
    parser.add_argument(
        "--no-fp16", action="store_true", help="Disable mixed precision on CUDA."
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU training (very slow)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before training.",
    )
    return parser.parse_args()


def subsample_positions(
    positions: np.ndarray,
    labels: np.ndarray,
    limit: int,
    seed: int,
) -> np.ndarray:
    """Take a stratified subset of row positions."""
    if limit <= 0 or limit >= len(positions):
        return positions

    subset, _ = train_test_split(
        positions,
        train_size=limit,
        random_state=seed,
        stratify=labels[positions],
    )
    return np.sort(subset)


def package_versions():
    """Record the exact stack used for the run."""
    import accelerate
    import datasets
    import torch
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "accelerate": accelerate.__version__,
    }


def numeric_metrics(metrics):
    """Convert Trainer metric values to floats and drop non-numeric fields."""
    converted = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        try:
            converted[key] = float(value)
        except (TypeError, ValueError):
            continue
    return converted


def main():
    args = parse_args()
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        TrainingArguments,
    )

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"Output directory is not empty: {output_dir}\n"
                "Pass --overwrite to replace it, or choose another --output-dir."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning("CUDA is unavailable; training will run on CPU.")

    logger.info("Loading %s", PREPROCESSED_DATA_PATH)
    data = load_preprocessed_data(PREPROCESSED_DATA_PATH)
    if TEXT_COLUMN not in data.columns:
        raise ValueError(f"Missing {TEXT_COLUMN!r} in {PREPROCESSED_DATA_PATH}")

    labels = data["severity_category"].astype(str).to_numpy()
    unknown = sorted(set(labels) - set(SEVERITY_LABELS))
    if unknown:
        raise ValueError(f"Unknown severity labels: {unknown}")

    train_positions, val_positions, test_positions = stratified_split_indices(
        labels,
        test_size=TEST_SIZE,
        val_size=args.val_size,
        random_state=args.seed,
    )
    train_positions = subsample_positions(
        train_positions, labels, args.max_train_samples, args.seed
    )
    val_positions = subsample_positions(
        val_positions, labels, args.max_val_samples, args.seed
    )

    save_split_indices(
        output_dir / "split_indices.json",
        train_positions,
        val_positions,
        test_positions,
    )
    (output_dir / "label_mapping.json").write_text(
        json.dumps(
            {
                "labels": SEVERITY_LABELS,
                "label_to_id": LABEL_TO_ID,
                "id_to_label": {
                    str(index): label for index, label in enumerate(SEVERITY_LABELS)
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train_texts = data.iloc[train_positions][TEXT_COLUMN].astype(str).tolist()
    val_texts = data.iloc[val_positions][TEXT_COLUMN].astype(str).tolist()
    train_labels = labels_to_ids(labels[train_positions])
    val_labels = labels_to_ids(labels[val_positions])

    if len(np.unique(train_labels)) != NUM_LABELS:
        raise ValueError("Training subset does not contain all six classes")

    logger.info(
        "Rows: train=%d validation=%d test=%d (test is not loaded for training)",
        len(train_positions),
        len(val_positions),
        len(test_positions),
    )
    logger.info(
        "Train distribution: %s",
        pd.Series(labels[train_positions]).value_counts().to_dict(),
    )
    logger.info(
        "Validation distribution: %s",
        pd.Series(labels[val_positions]).value_counts().to_dict(),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = BugTextDataset(
        train_texts, train_labels, tokenizer, max_length=args.max_length
    )
    val_dataset = BugTextDataset(
        val_texts, val_labels, tokenizer, max_length=args.max_length
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS,
        id2label={index: label for index, label in enumerate(SEVERITY_LABELS)},
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )

    class_weights = compute_class_weights(train_labels)
    logger.info(
        "Class weights: %s",
        {
            label: round(float(weight), 4)
            for label, weight in zip(SEVERITY_LABELS, class_weights)
        },
    )

    fp16 = use_cuda and not args.no_fp16
    if fp16:
        logger.info("Using fp16 mixed precision")
    else:
        logger.info("Using fp32")

    max_steps = args.max_steps if args.max_steps > 0 else -1
    eval_strategy = "steps" if max_steps > 0 else "epoch"
    eval_steps = max(max_steps, 1) if max_steps > 0 else None

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        warmup_steps=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=fp16,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=eval_strategy,
        save_steps=eval_steps or 500,
        logging_steps=10 if max_steps > 0 else 50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        use_cpu=not use_cuda,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ],
    )

    started = time.time()
    train_result = trainer.train()
    elapsed_minutes = (time.time() - started) / 60
    logger.info("Training finished in %.1f minutes", elapsed_minutes)

    eval_metrics = trainer.evaluate()
    logger.info("Best validation metrics: %s", eval_metrics)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    validation_metrics = numeric_metrics(eval_metrics)
    best_validation_f1 = numeric_metrics({"f1_macro": trainer.state.best_metric}).get(
        "f1_macro"
    )

    metadata = {
        "model_name": args.model_name,
        "text_column": TEXT_COLUMN,
        "labels": SEVERITY_LABELS,
        "label_to_id": LABEL_TO_ID,
        "max_length": args.max_length,
        "seed": args.seed,
        "test_size": TEST_SIZE,
        "val_size": args.val_size,
        "train_rows": int(len(train_positions)),
        "validation_rows": int(len(val_positions)),
        "test_rows": int(len(test_positions)),
        "test_set_evaluated": False,
        "class_weights": {
            label: float(weight)
            for label, weight in zip(SEVERITY_LABELS, class_weights)
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "label_smoothing": args.label_smoothing,
            "fp16": fp16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "early_stopping_patience": args.early_stopping_patience,
            "max_steps": max_steps,
        },
        "training": {
            "runtime_minutes": round(elapsed_minutes, 2),
            "train_loss": numeric_metrics({"loss": train_result.training_loss}).get(
                "loss"
            ),
            "best_validation_f1_macro": best_validation_f1,
        },
        "validation_metrics": validation_metrics,
        "versions": package_versions(),
        "device": (torch.cuda.get_device_name(0) if use_cuda else "cpu"),
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    summary_path = RESULTS_DIR / "transformer_training_summary.txt"
    summary_lines = [
        "TRANSFORMER TRAINING SUMMARY",
        "=" * 70,
        "",
        f"Model: {args.model_name}",
        f"Output: {output_dir}",
        f"Device: {metadata['device']}",
        f"Rows: train={len(train_positions)} validation={len(val_positions)} "
        f"test={len(test_positions)} (test not evaluated)",
        f"Max length: {args.max_length}",
        f"Runtime: {elapsed_minutes:.1f} minutes",
        f"Best validation macro F1: {best_validation_f1}",
        "",
        "Validation metrics:",
        json.dumps(metadata["validation_metrics"], indent=2),
        "",
        "The test set has NOT been evaluated.",
        "Run scripts/07_transformer_evaluation.py exactly once after reviewing "
        "the validation result.",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("Saved training summary to %s", summary_path)


if __name__ == "__main__":
    main()
