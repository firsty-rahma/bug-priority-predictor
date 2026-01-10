"""
Tests for visualization module.
"""
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.visualization import EvaluationVisualizer


class TestEvaluationVisualizer:
    """Test EvaluationVisualizer class."""
    
    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance."""
        class_labels = ['blocker', 'critical', 'major', 'minor', 'normal', 'trivial']
        return EvaluationVisualizer(class_labels=class_labels)
    
    @pytest.fixture
    def sample_error_data(self):
        """Sample error analysis data."""
        return pd.DataFrame({
            'Class': ['blocker', 'critical', 'major', 'minor', 'normal', 'trivial'],
            'Total': [10, 20, 30, 15, 100, 10],
            'Errors': [9, 10, 25, 13, 20, 8],
            'Error Rate (%)': [90, 50, 83, 87, 20, 80],
            'Accuracy (%)': [10, 50, 17, 13, 80, 20]
        })
    
    @pytest.fixture
    def sample_confusion_patterns(self):
        """Sample confusion patterns."""
        return pd.DataFrame({
            'true_label': ['normal', 'major', 'minor'],
            'predicted_label': ['major', 'normal', 'normal'],
            'count': [50, 30, 20]
        })
    
    def test_initialization(self, visualizer):
        """Test visualizer initialization."""
        assert visualizer is not None
        assert len(visualizer.class_labels) == 6
    
    def test_plot_confusion_matrix(self, visualizer, sample_predictions, temp_directory):
        """Test confusion matrix plotting."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        save_path = temp_directory / 'confusion_matrix.png'
        
        fig = visualizer.plot_confusion_matrix(
            y_true,
            y_pred,
            title='Test Confusion Matrix',
            save_path=save_path
        )
        
        # Check figure created
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check file saved
        assert save_path.exists()
        
        plt.close(fig)
    
    def test_plot_error_analysis(self, visualizer, sample_error_data, 
                                 sample_confusion_patterns, temp_directory):
        """Test error analysis plotting."""
        save_path = temp_directory / 'error_analysis.png'
        
        fig = visualizer.plot_error_analysis(
            sample_error_data,
            sample_confusion_patterns,
            save_path=save_path
        )
        
        # Check figure created
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check file saved
        assert save_path.exists()
        
        plt.close(fig)
    
    def test_plot_feature_importance(self, visualizer, temp_directory):
        """Test feature importance plotting."""
        feature_names = ['crash', 'component', 'product', 'error', 'hang']
        importances = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
        save_path = temp_directory / 'feature_importance.png'
        
        fig = visualizer.plot_feature_importance(
            feature_names,
            importances,
            top_n=5,
            save_path=save_path
        )
        
        # Check figure created
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check file saved
        assert save_path.exists()
        
        plt.close(fig)