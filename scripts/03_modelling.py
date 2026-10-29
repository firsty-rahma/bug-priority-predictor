#!/usr/bin/env python
"""
Model Training and Selection

Compares candidate pipelines using repeated cross-validation on the training
split and saves the winner. The held-out test set is never touched here; the
final single evaluation happens in 04_modelling_hypertuning.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)

from data.loader import load_preprocessed_data
from data.preprocessor import TextPreprocessor
from models.linear import TEXT_COLUMN
from models.train import ModelTrainer
from utils.config import (
    CV_FOLDS,
    CV_REPEATS,
    EVAL_SEEDS,
    MODEL_DIR,
    PREPROCESSED_DATA_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    RF_NGRAM_RANGE,
    RF_TFIDF_MAX_FEATURES,
    TEST_SIZE,
)
from utils.custom_transformers import FeatureCombiner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_features():
    """Load data and guarantee every column the candidates may need."""
    data = load_preprocessed_data(PREPROCESSED_DATA_PATH)

    if TEXT_COLUMN not in data.columns:
        preprocessor = TextPreprocessor()
        data[TEXT_COLUMN] = [
            preprocessor.build_model_text(short, long)
            for short, long in zip(data["short_description"], data["long_description"])
        ]

    if "text_length" not in data.columns:
        data["text_length"] = data["text_processed"].str.split().str.len()

    columns = [
        TEXT_COLUMN,
        "text_processed",
        "component_name",
        "product_name",
        "text_length",
    ]
    return data[columns].copy(), data["severity_category"].copy()


def build_candidates(trainer):
    """Return fresh, unfitted candidate estimators."""
    rf_baseline = trainer.create_pipeline(
        FeatureCombiner(
            max_features=RF_TFIDF_MAX_FEATURES,
            min_df=2,
            max_df=0.8,
            ngram_range=RF_NGRAM_RANGE,
        ),
        RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        use_smote=True,
    )

    return {
        "rf_smote_baseline": rf_baseline,
        "ovr_logistic_regression": trainer.create_linear_pipeline(multiclass="ovr"),
        "multinomial_logistic_regression": trainer.create_linear_pipeline(
            multiclass="multinomial"
        ),
    }


def cross_validate(estimator, X, y):
    """Mean and std across repeated stratified folds."""
    cv = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
    )
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
    return float(scores.mean()), float(scores.std())


def repeated_split_check(candidates, X, y_encoded, seeds):
    """Fit each candidate on several random splits.

    Diagnostic only: these scores are not used to select the winner.
    """
    rows = []
    for name, estimator in candidates.items():
        scores = []
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_encoded,
                test_size=TEST_SIZE,
                random_state=seed,
                stratify=y_encoded,
            )
            model = clone(estimator).fit(X_train, y_train)
            scores.append(f1_score(y_test, model.predict(X_test), average="macro"))
        rows.append(
            {
                "candidate": name,
                "mean_f1_macro": float(np.mean(scores)),
                "std_f1_macro": float(np.std(scores)),
                "scores": " ".join(f"{score:.4f}" for score in scores),
            }
        )
    return pd.DataFrame(rows)


def main():
    logger.info("=" * 70)
    logger.info("MODEL TRAINING AND SELECTION")
    logger.info("=" * 70)

    X, y = load_features()
    trainer = ModelTrainer(random_state=RANDOM_STATE)
    X_train, _, y_train, _ = trainer.prepare_data(X, y, test_size=TEST_SIZE)

    candidates = build_candidates(trainer)
    comparison = []

    for name, estimator in candidates.items():
        logger.info(f"\n{'-' * 70}")
        logger.info(f"Candidate: {name}")
        logger.info("-" * 70)

        started = time.time()
        cv_mean, cv_std = cross_validate(estimator, X_train, y_train)
        elapsed = time.time() - started

        comparison.append(
            {
                "candidate": name,
                "cv_f1_macro": cv_mean,
                "cv_f1_macro_std": cv_std,
                "cv_seconds": round(elapsed, 2),
            }
        )

        logger.info(f"CV F1-macro: {cv_mean:.4f} (+/-{cv_std:.4f}) | {elapsed:.1f}s")

    comparison_df = pd.DataFrame(comparison).sort_values("cv_f1_macro", ascending=False)
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    best_row = comparison_df.iloc[0]
    best_name = best_row["candidate"]
    best_model = clone(candidates[best_name]).fit(X_train, y_train)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"BEST MODEL (selected by CV): {best_name}")
    logger.info(f"CV F1-macro: {best_row['cv_f1_macro']:.4f}")
    logger.info("=" * 70)

    text_column = "text_processed" if best_name.startswith("rf_") else TEXT_COLUMN
    metadata = {
        "candidate": best_name,
        "selection_metric": "mean 2x5-fold CV F1-macro",
        "cv_f1_macro": float(best_row["cv_f1_macro"]),
        "cv_f1_macro_std": float(best_row["cv_f1_macro_std"]),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_size": len(X_train),
        "text_column": text_column,
    }
    model_path = MODEL_DIR / "best_model.pkl"
    trainer.save_model(best_model, model_path, metadata=metadata)

    logger.info("\nRobustness check (diagnostic only, not used for selection)")
    diagnostic = repeated_split_check(
        candidates, X, trainer.label_encoder.transform(y), EVAL_SEEDS
    )
    diagnostic.to_csv(RESULTS_DIR / "repeated_split_check.csv", index=False)
    logger.info("\n" + diagnostic.to_string(index=False))

    summary_path = RESULTS_DIR / "model_selection_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("MODEL SELECTION SUMMARY\n")
        handle.write("=" * 70 + "\n")
        handle.write(f"Date: {metadata['trained_at']}\n")
        handle.write(f"Selected by: {metadata['selection_metric']}\n")
        handle.write(f"Winner: {best_name}\n")
        handle.write(f"CV F1-macro: {metadata['cv_f1_macro']:.4f}\n\n")
        handle.write("CANDIDATE COMPARISON (sorted by CV)\n")
        handle.write("-" * 70 + "\n")
        handle.write(comparison_df.to_string(index=False))
        handle.write("\n\nThe held-out test set is evaluated once by 04.\n")

    logger.info(f"\nModel saved to: {model_path}")
    logger.info(f"Comparison saved to: {comparison_path}")
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
