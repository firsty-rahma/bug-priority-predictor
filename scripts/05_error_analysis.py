#!/usr/bin/env python
"""
Error Analysis Script

Performs comprehensive error analysis and feature importance analysis.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import logging

from utils.config import (
    PREPROCESSED_DATA_PATH,
    MODEL_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    TEST_SIZE,
    RANDOM_STATE,
)
from data.loader import load_preprocessed_data
from models.train import ModelTrainer
from evaluation.metrics import ModelEvaluator
from evaluation.visualization import EvaluationVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_feature_importance(model, X_train):
    """Extract and analyze feature importance."""
    logger.info("Analyzing feature importance...")
    
    # Get feature names
    feature_combiner = model.named_steps['feature_combiner']
    
    # TF-IDF feature names
    tfidf_features = feature_combiner.tfidf.get_feature_names_out()
    
    # Categorical features
    categorical_features = ['component', 'product', 'text_length']
    
    # Combine all feature names
    all_features = list(tfidf_features) + categorical_features
    
    # Get feature importance from Random Forest
    classifier = model.named_steps['classifier']
    importances = classifier.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Calculate feature type percentages
    n_tfidf = len(tfidf_features)
    tfidf_importance = importance_df.head(n_tfidf)['importance'].sum()
    categorical_importance = importance_df.tail(len(categorical_features))['importance'].sum()
    
    logger.info(f"\nFeature Importance by Type:")
    logger.info(f"  Text features (TF-IDF): {tfidf_importance*100:.1f}%")
    logger.info(f"  Categorical/Numeric: {categorical_importance*100:.1f}%")
    
    logger.info(f"\nTop 10 Most Important Features:")
    logger.info(importance_df.head(10).to_string(index=False))
    
    return importance_df, all_features, importances


def save_error_report(error_df, confusion_patterns, critical_errors, 
                     confidence_analysis, text_length_analysis, output_path):
    """Save comprehensive error analysis report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ERROR ANALYSIS REPORT\n")
        f.write("Bug Severity Classification - Random Forest Model\n")
        f.write("="*70 + "\n\n")
        
        # Overall statistics
        f.write("1. OVERALL STATISTICS\n")
        f.write("-"*70 + "\n")
        total = error_df['Total'].sum()
        errors = error_df['Errors'].sum()
        f.write(f"Total predictions: {total}\n")
        f.write(f"Correct: {total - errors} ({(total-errors)/total*100:.1f}%)\n")
        f.write(f"Incorrect: {errors} ({errors/total*100:.1f}%)\n\n")
        
        # Error rate by class
        f.write("2. ERROR RATE BY CLASS\n")
        f.write("-"*70 + "\n")
        f.write(error_df.to_string(index=False))
        f.write("\n\n")
        
        # Top misclassification patterns
        f.write("3. TOP MISCLASSIFICATION PATTERNS\n")
        f.write("-"*70 + "\n")
        f.write(confusion_patterns.to_string(index=False))
        f.write("\n\n")
        
        # Confidence analysis
        if confidence_analysis:
            f.write("4. CONFIDENCE ANALYSIS\n")
            f.write("-"*70 + "\n")
            f.write(f"Avg confidence (correct): {confidence_analysis['correct_avg']:.3f}\n")
            f.write(f"Avg confidence (incorrect): {confidence_analysis['incorrect_avg']:.3f}\n")
            f.write(f"Confidence gap: {confidence_analysis['gap']:.3f}\n\n")
        
        # Critical errors
        f.write("5. CRITICAL ERRORS\n")
        f.write("-"*70 + "\n")
        f.write(f"Critical/Blocker missed (predicted as normal): {critical_errors['critical_missed']}\n")
        f.write(f"False alarms (normal predicted as critical): {critical_errors['false_alarms']}\n\n")
        
        # Text length analysis
        if text_length_analysis is not None:
            f.write("6. TEXT LENGTH ANALYSIS\n")
            f.write("-"*70 + "\n")
            f.write(text_length_analysis.to_string())
            f.write("\n\n")
    
    logger.info(f"✅ Error report saved: {output_path}")


def main():
    """Main execution function."""
    logger.info("="*70)
    logger.info("ERROR ANALYSIS & FEATURE IMPORTANCE")
    logger.info("="*70)
    
    # Load data
    data = load_preprocessed_data(PREPROCESSED_DATA_PATH)
    
    # Add text length if not present
    if 'text_length' not in data.columns:
        data['text_length'] = data['text_processed'].str.split().str.len()
    
    # Prepare features
    X = data[['text_processed', 'component_name', 'product_name', 'text_length']].copy()
    y = data['severity_category']
    
    # Load trained model
    model_path = MODEL_DIR / "best_model_random_forest_tuned.pkl"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Please run 04_hyperparameter_tuning.py first")
        return
    
    logger.info(f"Loading model from {model_path}")
    model_data = ModelTrainer.load_model(model_path)
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    
    # Split data (same as training)
    from sklearn.model_selection import train_test_split
    y_encoded = label_encoder.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )
    
    # Predictions
    logger.info("\nGenerating predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Decode labels
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(class_labels=sorted(label_encoder.classes_))
    
    # Error analysis
    logger.info("\n" + "="*70)
    logger.info("PERFORMING ERROR ANALYSIS")
    logger.info("="*70)
    
    error_df = evaluator.error_analysis(y_test_decoded, y_pred_decoded)
    confusion_patterns = evaluator.confusion_patterns(y_test_decoded, y_pred_decoded, top_n=10)
    critical_errors = evaluator.critical_errors(y_test_decoded, y_pred_decoded)
    
    # Confidence analysis
    confidence_correct = y_pred_proba[y_test == y_pred].max(axis=1)
    confidence_incorrect = y_pred_proba[y_test != y_pred].max(axis=1)
    
    confidence_analysis = {
        'correct_avg': confidence_correct.mean(),
        'incorrect_avg': confidence_incorrect.mean(),
        'gap': confidence_correct.mean() - confidence_incorrect.mean()
    }
    
    logger.info(f"\nConfidence Analysis:")
    logger.info(f"  Correct predictions: {confidence_analysis['correct_avg']:.3f}")
    logger.info(f"  Incorrect predictions: {confidence_analysis['incorrect_avg']:.3f}")
    logger.info(f"  Gap: {confidence_analysis['gap']:.3f}")
    
    # Text length analysis
    test_data = X_test.copy()
    test_data['true_label'] = y_test_decoded
    test_data['pred_label'] = y_pred_decoded
    test_data['correct'] = y_test_decoded == y_pred_decoded
    
    # Bin text lengths
    test_data['text_length_bin'] = pd.cut(
        test_data['text_length'],
        bins=[0, 20, 50, 100, 200, np.inf],
        labels=['Very Short (0-20)', 'Short (20-50)', 'Medium (50-100)', 
                'Long (100-200)', 'Very Long (200+)']
    )
    
    text_length_analysis = test_data.groupby('text_length_bin').agg({
        'correct': ['count', lambda x: (~x).sum()]
    })
    text_length_analysis.columns = ['Total', 'Errors']
    text_length_analysis['Error Rate (%)'] = (
        text_length_analysis['Errors'] / text_length_analysis['Total'] * 100
    ).round(1)
    
    logger.info(f"\nText Length Analysis:")
    logger.info(text_length_analysis.to_string())
    
    # Feature importance
    logger.info("\n" + "="*70)
    logger.info("ANALYZING FEATURE IMPORTANCE")
    logger.info("="*70)
    
    importance_df, feature_names, importances = analyze_feature_importance(model, X_train)
    
    # Save results
    logger.info("\nSaving results...")
    
    # Save error report
    save_error_report(
        error_df,
        confusion_patterns,
        critical_errors,
        confidence_analysis,
        text_length_analysis,
        RESULTS_DIR / 'error_analysis_report.txt'
    )
    
    # Save misclassified cases
    misclassified = test_data[~test_data['correct']].copy()
    misclassified_export = misclassified[[
        'true_label', 'pred_label', 'text_length', 
        'text_processed', 'component_name', 'product_name'
    ]].copy()
    
    # Add confidence
    misclassified_indices = np.where(y_test != y_pred)[0]
    misclassified_export['confidence'] = y_pred_proba[misclassified_indices].max(axis=1)
    
    misclassified_path = RESULTS_DIR / 'misclassified_cases.csv'
    misclassified_export.to_csv(misclassified_path, index=False)
    logger.info(f"✅ Misclassified cases saved: {misclassified_path}")
    
    # Visualizations
    logger.info("\nCreating visualizations...")
    visualizer = EvaluationVisualizer(class_labels=sorted(label_encoder.classes_))
    
    # Error analysis plots
    visualizer.plot_error_analysis(
        error_df,
        confusion_patterns,
        save_path=FIGURES_DIR / 'error_analysis_overview.png'
    )
    
    # Feature importance
    visualizer.plot_feature_importance(
        feature_names,
        importances,
        top_n=30,
        save_path=FIGURES_DIR / 'feature_importance_detailed.png'
    )
    
    logger.info("\n" + "="*70)
    logger.info("✅ ERROR ANALYSIS COMPLETE")
    logger.info("="*70)
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
