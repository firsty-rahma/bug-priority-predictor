#!/usr/bin/env python
"""
Hyperparameter Tuning

Tunes the linear model by cross-validation on the training split.
The test set is evaluated once, after the configuration is chosen.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from data.loader import load_preprocessed_data
from data.preprocessor import TextPreprocessor
from models.linear import (
    MULTICLASS_C_PARAM,
    TEXT_COLUMN,
    build_linear_pipeline,
    linear_param_grid,
)
from models.train import ModelTrainer
from utils.config import (
    CV_FOLDS,
    CV_REPEATS,
    CV_TOLERANCE,
    DEFAULT_C,
    MODEL_DIR,
    PREPROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_NGRAM_RANGE,
    RAW_TFIDF_MAX_FEATURES,
    RESULTS_DIR,
    TEST_SIZE,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_features():
    """Load data and guarantee the near-raw text column."""
    data = load_preprocessed_data(PREPROCESSED_DATA_PATH)

    if TEXT_COLUMN not in data.columns:
        preprocessor = TextPreprocessor()
        data[TEXT_COLUMN] = [
            preprocessor.build_model_text(short, long)
            for short, long in zip(data["short_description"], data["long_description"])
        ]

    return data[[TEXT_COLUMN]].copy(), data["severity_category"].copy()


def tune(strategy, X_train, y_train):
    """Grid-search one multiclass strategy on the training split."""
    pipeline = build_linear_pipeline(multiclass=strategy, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline,
        linear_param_grid(strategy),
        cv=RepeatedStratifiedKFold(
            n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
        ),
        scoring="f1_macro",
        n_jobs=1,
        verbose=1,
        error_score="raise",
    )
    started = time.time()
    search.fit(X_train, y_train)
    return search, time.time() - started


def choose_params(search, strategy):
    """Choose the winning configuration with a tolerance for noise.

    Compare against the full default configuration, not just the default
    ``C``: otherwise a tuned n-gram range can ride along with the default
    regularisation and win on a difference that is inside fold noise.

    If the best configuration beats the default by less than CV_TOLERANCE,
    keep the default: tiny CV differences do not replicate on held-out data.
    """
    results = search.cv_results_
    scores = results["mean_test_score"]

    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])

    reference = {
        MULTICLASS_C_PARAM[strategy]: DEFAULT_C,
        "features__word__max_features": RAW_TFIDF_MAX_FEATURES,
        "features__word__ngram_range": RAW_NGRAM_RANGE,
    }

    reference_rows = [
        index
        for index, params in enumerate(results["params"])
        if all(
            (
                tuple(params[key]) == value
                if isinstance(value, tuple)
                else params[key] == value
            )
            for key, value in reference.items()
        )
    ]
    if reference_rows:
        reference_index = reference_rows[0]
        reference_score = float(scores[reference_index])
        if best_score - reference_score < CV_TOLERANCE:
            return dict(results["params"][reference_index]), reference_score

    return dict(results["params"][best_index]), best_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=["ovr", "multinomial", "both"],
        default="both",
        help="Multiclass strategy to tune (default: both).",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("HYPERPARAMETER TUNING")
    logger.info("=" * 70)

    X, y = load_features()
    trainer = ModelTrainer(random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=TEST_SIZE)

    strategies = ["ovr", "multinomial"] if args.strategy == "both" else [args.strategy]

    results = {}
    for strategy in strategies:
        logger.info(f"\nTuning: {strategy}")
        search, elapsed = tune(strategy, X_train, y_train)
        params, score = choose_params(search, strategy)
        results[strategy] = {
            "search": search,
            "params": params,
            "score": score,
            "time": elapsed,
        }
        logger.info(f"{strategy}: CV F1-macro {score:.4f} in {elapsed / 60:.2f} min")

    best_strategy = max(results, key=lambda key: results[key]["score"])
    best_params = results[best_strategy]["params"]
    best_model = build_linear_pipeline(
        multiclass=best_strategy, random_state=RANDOM_STATE
    ).set_params(**best_params)
    best_model.fit(X_train, y_train)

    # Single test evaluation, after selection
    eval_results = trainer.evaluate(best_model, X_test, y_test)

    metadata = {
        "candidate": f"{best_strategy}_logistic_regression_tuned",
        "selection_metric": "mean 2x5-fold CV F1-macro",
        "cv_f1_macro": float(results[best_strategy]["score"]),
        "test_f1_macro": float(eval_results["f1_macro"]),
        "test_f1_weighted": float(eval_results["f1_weighted"]),
        "best_params": {key: str(value) for key, value in best_params.items()},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "text_column": TEXT_COLUMN,
    }
    model_path = MODEL_DIR / "best_model.pkl"
    trainer.save_model(best_model, model_path, metadata=metadata)

    summary_path = RESULTS_DIR / "hyperparameter_tuning_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("HYPERPARAMETER TUNING SUMMARY\n")
        handle.write("=" * 70 + "\n\n")
        handle.write(f"Date: {metadata['trained_at']}\n")
        handle.write(f"Dataset size: {len(X)} rows\n")
        handle.write(f"Train set: {len(X_train)} samples\n")
        handle.write(f"Test set: {len(X_test)} samples\n\n")

        for strategy, result in results.items():
            handle.write(f"{strategy}:\n")
            handle.write(f"  Best CV F1-macro: {result['score']:.4f}\n")
            handle.write(f"  Tuning time: {result['time'] / 60:.2f} minutes\n")
            handle.write("  Best parameters:\n")
            for key, value in result["params"].items():
                handle.write(f"    {key}: {value}\n")
            handle.write("\n")

        handle.write("=" * 70 + "\n")
        handle.write(f"BEST: {best_strategy}\n")
        handle.write(f"CV F1-macro: {metadata['cv_f1_macro']:.4f}\n")
        handle.write(f"Test F1-macro: {metadata['test_f1_macro']:.4f}\n")
        handle.write("=" * 70 + "\n\n")
        handle.write("TEST SET PERFORMANCE (single evaluation):\n")
        handle.write(
            pd.DataFrame(eval_results["classification_report"]).transpose().to_string()
        )

    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
