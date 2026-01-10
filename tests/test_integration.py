"""
Integration tests for end-to-end workflows.
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.preprocessor import TextPreprocessor
from utils.custom_transformers import FeatureCombiner
from models.train import ModelTrainer
from sklearn.ensemble import RandomForestClassifier


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_pipeline_flow(self, sample_bug_data, temp_directory):
        """Test complete pipeline from raw data to prediction."""
        # Step 1: Preprocess
        preprocessor = TextPreprocessor(use_pos_lemmatization=False)  # Faster for tests
        data_processed = preprocessor.preprocess_dataframe(sample_bug_data, show_progress=False)
        
        assert 'text_processed' in data_processed.columns
        assert len(data_processed) > 0
        
        # Step 2: Prepare features
        X = data_processed[['text_processed', 'component_name', 
                           'product_name', 'text_length']]
        y = data_processed['severity_category']
        
        # Step 3: Train model
        trainer = ModelTrainer(random_state=42)
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            X, y, test_size=0.3  # Larger test size for small data
        )
        
        feature_combiner = FeatureCombiner(
            max_features=10,
            min_df=1,
            max_df=1.0
        )
        classifier = RandomForestClassifier(n_estimators=10, random_state=42)
        pipeline = trainer.create_pipeline(
            feature_combiner, 
            classifier, 
            use_smote=False  # Disabled for test
        )
        
        # Fit
        pipeline.fit(X_train, y_train)
        
        # Step 4: Predict
        predictions = pipeline.predict(X_test)
        
        assert len(predictions) == len(X_test)
        
        # Step 5: Save model
        model_path = temp_directory / 'integration_test_model.pkl'
        trainer.save_model(pipeline, model_path)
        
        assert model_path.exists()
        
        # Step 6: Load and predict again
        loaded_model_data = ModelTrainer.load_model(model_path)
        predictions_loaded = loaded_model_data['model'].predict(X_test)
        
        # Predictions should be identical
        assert (predictions == predictions_loaded).all()
    
    def test_preprocessing_to_features(self, sample_bug_data):
        """Test preprocessing followed by feature engineering."""
        # Preprocess (without POS for speed)
        preprocessor = TextPreprocessor(
            custom_stopwords=['test', 'bug'],
            use_pos_lemmatization=False
        )
        data_processed = preprocessor.preprocess_dataframe(
            sample_bug_data, 
            show_progress=False
        )
        
        # Feature engineering
        X = data_processed[['text_processed', 'component_name', 
                           'product_name', 'text_length']]
        
        combiner = FeatureCombiner(
            max_features=20,
            min_df=1,
            max_df=1.0
        )
        combiner.fit(X)
        X_transformed = combiner.transform(X)
        
        # Check output
        assert X_transformed.shape[0] == len(data_processed)
        assert X_transformed.shape[1] > 3  # At least some TF-IDF features