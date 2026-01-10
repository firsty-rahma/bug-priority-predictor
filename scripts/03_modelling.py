#!/usr/bin/env python
"""
Model Training Script

Trains bug severity classification models with SMOTE.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import logging

from utils.config import (
    PREPROCESSED_DATA_PATH,
    MODEL_DIR,
    FIGURES_DIR,
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
    SEVERITY_CLASSES
)
from data.loader import load_preprocessed_data
from utils.custom_transformers import FeatureCombiner
from models.train import ModelTrainer
from evaluation.metrics import ModelEvaluator
from evaluation.visualization import EvaluationVisualizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    logger.info("="*70)
    logger.info("MODEL TRAINING")
    logger.info("="*70)
    
    # Load data
    data = load_preprocessed_data(PREPROCESSED_DATA_PATH)
    
    # Add text length if not present
    if 'text_length' not in data.columns:
        data['text_length'] = data['text_processed'].str.split().str.len()
    
    # Prepare features
    X = data[['text_processed', 'component_name', 'product_name', 'text_length']].copy()
    y = data['severity_category']
    
    logger.info(f"\nClass distribution:")
    logger.info(y.value_counts().sort_index())
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=RANDOM_STATE)
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=TEST_SIZE)
    
    # Define models to compare
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    }
    
    results = {}
    
    # Train and evaluate each model
    for model_name, classifier in models.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"Training: {model_name}")
        logger.info("="*70)
        
        # Create pipeline
        feature_combiner = FeatureCombiner(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        pipeline = trainer.create_pipeline(
            feature_combiner,
            classifier,
            use_smote=True
        )
        
        # Cross-validation
        cv_results = trainer.train_with_cv(pipeline, X_train, y_train, cv_folds=CV_FOLDS)
        
        # Train on full training set
        pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        eval_results = trainer.evaluate(pipeline, X_test, y_test)
        
        results[model_name] = {
            'cv_score': cv_results['mean'],
            'cv_std': cv_results['std'],
            'test_f1_macro': eval_results['f1_macro'],
            'test_f1_weighted': eval_results['f1_weighted'],
            'model': pipeline
        }
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['test_f1_macro'])
    best_model = results[best_model_name]['model']
    
    logger.info(f"\n{'='*70}")
    logger.info(f"BEST MODEL: {best_model_name}")
    logger.info(f"Test F1-Macro: {results[best_model_name]['test_f1_macro']:.4f}")
    logger.info("="*70)
    
    # Save best model
    model_path = MODEL_DIR / "best_model_baseline.pkl"
    trainer.save_model(best_model, model_path)
    
    # Detailed evaluation
    logger.info("\nPerforming detailed evaluation...")
    evaluator = ModelEvaluator(class_labels=sorted(trainer.label_encoder.classes_))
    
    # Get predictions
    y_pred = best_model.predict(X_test)
    y_test_decoded = trainer.label_encoder.inverse_transform(y_test)
    y_pred_decoded = trainer.label_encoder.inverse_transform(y_pred)
    
    # Error analysis
    error_df = evaluator.error_analysis(y_test_decoded, y_pred_decoded)
    confusion_patterns = evaluator.confusion_patterns(y_test_decoded, y_pred_decoded)
    critical_errors = evaluator.critical_errors(y_test_decoded, y_pred_decoded)
    
    # Visualizations
    logger.info("\nCreating visualizations...")
    visualizer = EvaluationVisualizer(class_labels=sorted(trainer.label_encoder.classes_))
    
    # Confusion matrix
    visualizer.plot_confusion_matrix(
        y_test_decoded,
        y_pred_decoded,
        title=f'Confusion Matrix - {best_model_name}',
        save_path=FIGURES_DIR / 'confusion_matrix_baseline.png'
    )
    
    # Error analysis plots
    visualizer.plot_error_analysis(
        error_df,
        confusion_patterns,
        save_path=FIGURES_DIR / 'error_analysis_baseline.png'
    )
    
    logger.info("\n" + "="*70)
    logger.info("✅ MODEL TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Best model saved to: {model_path}")
    logger.info(f"Visualizations saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()