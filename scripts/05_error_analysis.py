#!/usr/bin/env python
"""
Error Analysis Script

Performs comprehensive error analysis and feature importance analysis for
the selected model. Supports both the linear pipeline and the legacy Random
Forest baseline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

import numpy as np
import pandas as pd

from data.loader import load_preprocessed_data
from data.preprocessor import TextPreprocessor
from evaluation.metrics import ModelEvaluator
from evaluation.visualization import EvaluationVisualizer
from models.linear import TEXT_COLUMN
from models.train import ModelTrainer
from utils.config import (
    BEST_MODEL_PATH,
    FIGURES_DIR,
    LEGACY_MODEL_PATH,
    PREPROCESSED_DATA_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_features():
    """Load data with every column either pipeline family may need."""
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


def analyze_linear_importance(model, label_encoder):
    """Aggregate absolute coefficients across classes and per class."""
    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]

    feature_names = np.asarray(features.get_feature_names_out())

    # Older sklearn versions forwarded coef_ from OneVsRestClassifier;
    # current versions expose the fitted binary estimators directly.
    if hasattr(classifier, "coef_"):
        coefficients = np.asarray(classifier.coef_)
    else:
        coefficients = np.vstack(
            [estimator.coef_[0] for estimator in classifier.estimators_]
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": np.mean(np.abs(coefficients), axis=0),
        }
    ).sort_values("importance", ascending=False)

    per_class = {}
    for index, label in enumerate(label_encoder.classes_):
        per_class[label] = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "coefficient": coefficients[index],
                }
            )
            .sort_values("coefficient", ascending=False)
            .head(15)
        )

    logger.info("\nTop 20 features (mean absolute coefficient):")
    logger.info(importance_df.head(20).to_string(index=False))

    return (
        importance_df,
        feature_names,
        importance_df["importance"].to_numpy(),
        per_class,
    )


def analyze_rf_importance(model):
    """Extract feature importances from the legacy Random Forest pipeline."""
    feature_combiner = model.named_steps["feature_combiner"]
    tfidf_features = list(feature_combiner.tfidf.get_feature_names_out())
    categorical_features = ["component", "product", "text_length"]

    feature_names = np.asarray(tfidf_features + categorical_features)
    importances = np.asarray(model.named_steps["classifier"].feature_importances_)

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    logger.info("\nTop 20 features:")
    logger.info(importance_df.head(20).to_string(index=False))

    return importance_df, feature_names, importances, {}


def analyze_feature_importance(model, label_encoder):
    """Dispatch to the analyzer matching the pipeline family."""
    if "feature_combiner" in model.named_steps:
        return analyze_rf_importance(model)
    return analyze_linear_importance(model, label_encoder)


def save_error_report(
    error_df,
    confusion_patterns,
    critical_errors,
    confidence_analysis,
    text_length_analysis,
    model_name,
    output_path,
):
    """Save comprehensive error analysis report."""
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("=" * 70 + "\n")
        handle.write("ERROR ANALYSIS REPORT\n")
        handle.write(f"Bug Severity Classification - {model_name}\n")
        handle.write("=" * 70 + "\n\n")

        handle.write("1. OVERALL STATISTICS\n")
        handle.write("-" * 70 + "\n")
        total = error_df["Total"].sum()
        errors = error_df["Errors"].sum()
        handle.write(f"Total predictions: {total}\n")
        handle.write(
            f"Correct: {total - errors} ({(total - errors) / total * 100:.1f}%)\n"
        )
        handle.write(f"Incorrect: {errors} ({errors / total * 100:.1f}%)\n\n")

        handle.write("2. ERROR RATE BY CLASS\n")
        handle.write("-" * 70 + "\n")
        handle.write(error_df.to_string(index=False))
        handle.write("\n\n")

        handle.write("3. TOP MISCLASSIFICATION PATTERNS\n")
        handle.write("-" * 70 + "\n")
        handle.write(confusion_patterns.to_string(index=False))
        handle.write("\n\n")

        if confidence_analysis:
            handle.write("4. CONFIDENCE ANALYSIS\n")
            handle.write("-" * 70 + "\n")
            handle.write(
                f"Avg confidence (correct): {confidence_analysis['correct_avg']:.3f}\n"
            )
            handle.write(
                f"Avg confidence (incorrect): {confidence_analysis['incorrect_avg']:.3f}\n"
            )
            handle.write(f"Confidence gap: {confidence_analysis['gap']:.3f}\n\n")

        handle.write("5. CRITICAL ERRORS\n")
        handle.write("-" * 70 + "\n")
        handle.write(
            "Critical/Blocker missed (predicted as normal): "
            f"{critical_errors['critical_missed']}\n"
        )
        handle.write(
            f"False alarms (normal predicted as critical): "
            f"{critical_errors['false_alarms']}\n\n"
        )

        if text_length_analysis is not None:
            handle.write("6. TEXT LENGTH ANALYSIS\n")
            handle.write("-" * 70 + "\n")
            handle.write(text_length_analysis.to_string())
            handle.write("\n\n")

    logger.info(f"Error report saved: {output_path}")


def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("ERROR ANALYSIS & FEATURE IMPORTANCE")
    logger.info("=" * 70)

    model_path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else LEGACY_MODEL_PATH
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Please run 03_modelling.py first")
        return

    logger.info(f"Loading model from {model_path}")
    model_data = ModelTrainer.load_model(model_path)
    model = model_data["model"]
    label_encoder = model_data["label_encoder"]
    metadata = model_data.get("metadata", {})
    model_name = metadata.get("candidate", model_path.stem)

    X, y = load_features()
    y_encoded = label_encoder.transform(y)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    logger.info("\nGenerating predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    evaluator = ModelEvaluator(class_labels=sorted(label_encoder.classes_))

    logger.info("\n" + "=" * 70)
    logger.info("PERFORMING ERROR ANALYSIS")
    logger.info("=" * 70)

    error_df = evaluator.error_analysis(y_test_decoded, y_pred_decoded)
    confusion_patterns = evaluator.confusion_patterns(
        y_test_decoded, y_pred_decoded, top_n=10
    )
    critical_errors = evaluator.critical_errors(y_test_decoded, y_pred_decoded)

    confidence_correct = y_pred_proba[y_test == y_pred].max(axis=1)
    confidence_incorrect = y_pred_proba[y_test != y_pred].max(axis=1)

    confidence_analysis = {
        "correct_avg": confidence_correct.mean(),
        "incorrect_avg": confidence_incorrect.mean(),
        "gap": confidence_correct.mean() - confidence_incorrect.mean(),
    }

    logger.info("\nConfidence Analysis:")
    logger.info(f"  Correct predictions: {confidence_analysis['correct_avg']:.3f}")
    logger.info(f"  Incorrect predictions: {confidence_analysis['incorrect_avg']:.3f}")
    logger.info(f"  Gap: {confidence_analysis['gap']:.3f}")

    test_data = X_test.copy()
    test_data["true_label"] = y_test_decoded
    test_data["pred_label"] = y_pred_decoded
    test_data["correct"] = y_test_decoded == y_pred_decoded

    test_data["text_length_bin"] = pd.cut(
        test_data["text_length"],
        bins=[0, 20, 50, 100, 200, np.inf],
        labels=[
            "Very Short (0-20)",
            "Short (20-50)",
            "Medium (50-100)",
            "Long (100-200)",
            "Very Long (200+)",
        ],
    )

    text_length_analysis = test_data.groupby("text_length_bin", observed=False).agg(
        {"correct": ["count", lambda values: (~values).sum()]}
    )
    text_length_analysis.columns = ["Total", "Errors"]
    text_length_analysis["Error Rate (%)"] = (
        text_length_analysis["Errors"] / text_length_analysis["Total"] * 100
    ).round(1)

    logger.info("\nText Length Analysis:")
    logger.info(text_length_analysis.to_string())

    logger.info("\n" + "=" * 70)
    logger.info("ANALYZING FEATURE IMPORTANCE")
    logger.info("=" * 70)

    importance_df, feature_names, importances, per_class = analyze_feature_importance(
        model, label_encoder
    )

    logger.info("\nSaving results...")
    save_error_report(
        error_df,
        confusion_patterns,
        critical_errors,
        confidence_analysis,
        text_length_analysis,
        model_name,
        RESULTS_DIR / "error_analysis_report.txt",
    )

    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    if per_class:
        pd.concat(
            [frame.assign(severity=label) for label, frame in per_class.items()],
            ignore_index=True,
        ).to_csv(RESULTS_DIR / "top_features_by_class.csv", index=False)

    misclassified = test_data[~test_data["correct"]].copy()
    misclassified_export = misclassified[
        [
            "true_label",
            "pred_label",
            "text_length",
            TEXT_COLUMN,
            "component_name",
            "product_name",
        ]
    ].copy()
    misclassified_export["confidence"] = y_pred_proba[y_test != y_pred].max(axis=1)

    misclassified_path = RESULTS_DIR / "misclassified_cases.csv"
    misclassified_export.to_csv(misclassified_path, index=False)
    logger.info(f"Misclassified cases saved: {misclassified_path}")

    logger.info("\nCreating visualizations...")
    visualizer = EvaluationVisualizer(class_labels=sorted(label_encoder.classes_))

    visualizer.plot_confusion_matrix(
        y_test_decoded,
        y_pred_decoded,
        title=f"Confusion Matrix - {model_name}",
        save_path=FIGURES_DIR / "confusion_matrix_final.png",
    )

    visualizer.plot_error_analysis(
        error_df,
        confusion_patterns,
        save_path=FIGURES_DIR / "error_analysis_overview.png",
    )

    visualizer.plot_feature_importance(
        feature_names,
        importances,
        top_n=30,
        save_path=FIGURES_DIR / "feature_importance_detailed.png",
    )

    logger.info("\n" + "=" * 70)
    logger.info("ERROR ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
