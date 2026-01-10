"""
Tests for model training module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.train import ModelTrainer
from sklearn.ensemble import RandomForestClassifier
from utils.custom_transformers import FeatureCombiner


class TestModelTrainer:
    """Test ModelTrainer class."""
    
    def test_initialization(self):
        """Test trainer initialization."""
        trainer = ModelTrainer(random_state=42)
        
        assert trainer.random_state == 42
        assert trainer.label_encoder is not None
        assert trainer.best_model is None
    
    def test_prepare_data(self, preprocessed_bug_data):
        """Test data preparation."""
        trainer = ModelTrainer(random_state=42)
        
        X = preprocessed_bug_data[['text_processed', 'component_name', 
                                    'product_name', 'text_length']]
        y = preprocessed_bug_data['severity_category']
        
        X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=0.2)
        
        # Check split sizes
        total = len(X)
        assert len(X_train) == int(total * 0.8)
        assert len(X_test) == int(total * 0.2)
        
        # Check labels are encoded
        assert y_train.dtype in [np.int32, np.int64]
        assert y_test.dtype in [np.int32, np.int64]
    
    def test_create_pipeline(self):
        """Test pipeline creation."""
        trainer = ModelTrainer()
        
        feature_combiner = FeatureCombiner()
        classifier = RandomForestClassifier(random_state=42)
        
        # With SMOTE
        pipeline = trainer.create_pipeline(feature_combiner, classifier, use_smote=True)
        assert len(pipeline.steps) == 3
        assert pipeline.steps[1][0] == 'smote'
        
        # Without SMOTE
        pipeline = trainer.create_pipeline(feature_combiner, classifier, use_smote=False)
        assert len(pipeline.steps) == 2
        assert 'smote' not in [step[0] for step in pipeline.steps]
    
    def test_save_and_load_model(self, preprocessed_bug_data, temp_directory):
        """Test model saving and loading."""
        trainer = ModelTrainer(random_state=42)
        
        # Prepare data
        X = preprocessed_bug_data[['text_processed', 'component_name', 
                                    'product_name', 'text_length']]
        y = preprocessed_bug_data['severity_category']
        
        X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=0.2)
        
        # Create and train simple model
        feature_combiner = FeatureCombiner(max_features=10)
        classifier = RandomForestClassifier(n_estimators=10, random_state=42)
        pipeline = trainer.create_pipeline(feature_combiner, classifier, use_smote=False)
        pipeline.fit(X_train, y_train)
        
        # Save
        model_path = temp_directory / 'test_model.pkl'
        trainer.save_model(pipeline, model_path, include_encoder=True)
        assert model_path.exists()
        
        # Load
        loaded_data = ModelTrainer.load_model(model_path)
        assert 'model' in loaded_data
        assert 'label_encoder' in loaded_data
        
        # Test loaded model can predict
        predictions = loaded_data['model'].predict(X_test)
        assert len(predictions) == len(X_test)