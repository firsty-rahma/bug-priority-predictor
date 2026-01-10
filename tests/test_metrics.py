"""
Tests for evaluation metrics module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.metrics import ModelEvaluator


class TestModelEvaluator:
    """Test ModelEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        class_labels = ['blocker', 'critical', 'major', 'minor', 'normal', 'trivial']
        return ModelEvaluator(class_labels=class_labels)
    
    def test_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator is not None
        assert len(evaluator.class_labels) == 6
    
    def test_calculate_metrics(self, evaluator, sample_predictions):
        """Test metrics calculation."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        # Check required metrics exist
        assert 'f1_macro' in metrics
        assert 'f1_weighted' in metrics
        assert 'accuracy' in metrics
        
        # Check per-class metrics
        for label in evaluator.class_labels:
            assert f'{label}_precision' in metrics
            assert f'{label}_recall' in metrics
            assert f'{label}_f1' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['f1_macro'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_error_analysis(self, evaluator, sample_predictions):
        """Test error analysis."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        
        error_df = evaluator.error_analysis(y_true, y_pred)
        
        # Check DataFrame structure
        assert isinstance(error_df, pd.DataFrame)
        assert 'Class' in error_df.columns
        assert 'Total' in error_df.columns
        assert 'Errors' in error_df.columns
        assert 'Error Rate (%)' in error_df.columns
        
        # Check error rates
        assert (error_df['Error Rate (%)'] >= 0).all()
        assert (error_df['Error Rate (%)'] <= 100).all()
    
    def test_confusion_patterns(self, evaluator, sample_predictions):
        """Test confusion pattern identification."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        
        patterns = evaluator.confusion_patterns(y_true, y_pred, top_n=5)
        
        # Check DataFrame structure
        assert isinstance(patterns, pd.DataFrame)
        
        if len(patterns) > 0:
            assert 'true_label' in patterns.columns
            assert 'predicted_label' in patterns.columns
            assert 'count' in patterns.columns
            
            # Check sorted by count
            assert (patterns['count'].diff().dropna() <= 0).all()
    
    def test_critical_errors(self, evaluator):
        """Test critical error detection."""
        # Create test data with critical misses
        y_true = np.array(['blocker', 'critical', 'critical', 'normal', 'normal'])
        y_pred = np.array(['normal', 'normal', 'critical', 'critical', 'normal'])
        
        critical_errors = evaluator.critical_errors(y_true, y_pred)
        
        # Check structure
        assert 'critical_missed' in critical_errors
        assert 'false_alarms' in critical_errors
        assert 'ratio' in critical_errors
        
        # Verify counts
        # 2 critical/blocker predicted as normal
        assert critical_errors['critical_missed'] == 2
        
        # 1 normal predicted as critical
        assert critical_errors['false_alarms'] == 1