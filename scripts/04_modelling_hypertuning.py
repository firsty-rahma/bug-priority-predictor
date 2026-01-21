"""
Hyperparameter Tuning Script

Performs grid search to find optimal model hyperparameters.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import logging
import time

from utils.config import (
    PREPROCESSED_DATA_PATH,
    MODEL_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
    RF_PARAM_GRID,
    LR_PARAM_GRID,
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
    """Main execution function"""
    logger.info("="*70)
    logger.info("HYPERPARAMETER TUNING")
    logger.info("="*70)

    # Load data
    data = load_preprocessed_data(PREPROCESSED_DATA_PATH)

    # Add text length if not present
    if 'text_length' not in data.columns:
        data['text_length'] = data['text_processed'].str.split().str.len()

    # Prepare features
    X = data[['text_processed', 'component_name', 'product_name', 'text_length']].copy()
    y = data['severity_category']

    # Initialize trainer
    trainer = ModelTrainer(random_state=RANDOM_STATE)

    # Split data
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=TEST_SIZE)

    # Models to tune
    models_to_tune = {
        'Random Forest': {
            'classifier': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            'param_grid': RF_PARAM_GRID
        },
        'Logistic Regression': {
            'classifier': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            'param_grid': LR_PARAM_GRID
        }
    }

    tuning_results = {}

    # Tune each model
    for model_name, config in models_to_tune.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"Tuning: {model_name}")
        logger.info("="*70)

        start_time = time.time()

        # Create pipeline
        feature_combiner = FeatureCombiner(
            max_features=1000,
            min_df=2,
            max_df=0.8
        )
        
        pipeline = trainer.create_pipeline(
            feature_combiner,
            config['classifier'],
            use_smote=True
        )
        
        # Hyperparameter tuning
        grid_search = trainer.hyperparameter_tuning(
            pipeline,
            config['param_grid'],
            X_train,
            y_train,
            cv_folds=CV_FOLDS
        )
        
        elapsed_time = time.time() - start_time
        
        # Store results
        tuning_results[model_name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'grid_search': grid_search,
            'time': elapsed_time
        }
        
        logger.info(f"Tuning completed in {elapsed_time/60:.2f} minutes")

    # Find best overall model
    best_model_name = max(tuning_results, key=lambda x: tuning_results[x]['best_score'])
    best_grid_search = tuning_results[best_model_name]['grid_search']
    best_model = best_grid_search.best_estimator_
    best_score = tuning_results[best_model_name]['best_score']

    logger.info(f"\n{'='*70}")
    logger.info(f"BEST MODEL: {best_model_name}")
    logger.info(f"Best CV F1-Macro: {best_score:.4f}")
    logger.info(f"Best Parameters: {tuning_results[best_model_name]['best_params']}")
    logger.info("="*70)

    # Evaluate best model on test set
    logger.info("\nEvaluating best model on test set...")
    eval_results = trainer.evaluate(best_model, X_test, y_test)

    logger.info(f"\nTest Set Performance:")
    logger.info(f"  F1-Macro: {eval_results['f1_macro']:.4f}")
    logger.info(f"  F1-Weighted: {eval_results['f1_weighted']:.4f}")

    # Save best model
    model_path = MODEL_DIR / "best_model_random_forest_tuned.pkl"
    trainer.save_model(best_model, model_path)

    # Save tuning summary
    summary_path = RESULTS_DIR / "hyperparameter_tuning_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("HYPERPARAMETER TUNING SUMMARY\n")
        f.write("="*70 + "\n\n")

        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset size: {len(data)} rows\n")
        f.write(f"Train set: {len(X_train)} samples\n")
        f.write(f"Test set: {len(X_test)} samples\n\n")

        f.write("Label Encoding:\n")
        for idx, label in enumerate(trainer.label_encoder.classes_):
            f.write(f"  {label} -> {idx}\n")
        f.write("\n")
        
        f.write("-"*70 + "\n")
        f.write("MODEL COMPARISON\n")
        f.write("-"*70 + "\n\n")
        
        for model_name, results in tuning_results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Best CV F1-Macro: {results['best_score']:.4f}\n")
            f.write(f"  Tuning Time: {results['time']/60:.2f} minutes\n")
            f.write(f"  Best Parameters:\n")
            for param, value in results['best_params'].items():
                f.write(f"    {param}: {value}\n")
            f.write("\n")
        
        f.write("="*70 + "\n")
        f.write(f"BEST MODEL: {best_model_name}\n")
        f.write(f"CV F1-Macro: {best_score:.4f}\n")
        f.write(f"Test F1-Macro: {eval_results['f1_macro']:.4f}\n")
        f.write("="*70 + "\n\n")
        
        f.write("TEST SET PERFORMANCE:\n")
        from sklearn.metrics import classification_report
        f.write(classification_report(
            eval_results['y_test'],
            eval_results['y_pred']
        ))

    logger.info(f"✅ Tuning summary saved: {summary_path}")

    # Visualizations
    logger.info("\nCreating visualizations...")
    visualizer = EvaluationVisualizer(class_labels=sorted(trainer.label_encoder.classes_))

    visualizer.plot_confusion_matrix(
        eval_results['y_test'],
        eval_results['y_pred'],
        title='Confusion Matrix - Tuned Model',
        save_path=FIGURES_DIR / 'confusion_matrix_tuned_final.png'
    )

    logger.info("\n" + "="*70)
    logger.info("✅ HYPERPARAMETER TUNING COMPLETE")
    logger.info("="*70)
    logger.info(f"Best model saved to: {model_path}")
    logger.info(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()