"""
Model evaluation and analysis utilities.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support
)
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and error analysis."""
    
    def __init__(self, class_labels: List[str]):
        """
        Initialize evaluator.
        
        Parameters
        ----------
        class_labels : List[str]
            List of class labels in order
        """
        self.class_labels = class_labels
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
            
        Returns
        -------
        dict
            Dictionary of metrics
        """
        metrics = {
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'accuracy': (y_true == y_pred).mean()
        }
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.class_labels
        )
        
        for i, label in enumerate(self.class_labels):
            metrics[f'{label}_precision'] = precision[i]
            metrics[f'{label}_recall'] = recall[i]
            metrics[f'{label}_f1'] = f1[i]
            metrics[f'{label}_support'] = support[i]
        
        return metrics
    
    def error_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Perform detailed error analysis.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        confidence : np.ndarray, optional
            Prediction confidence scores
            
        Returns
        -------
        pd.DataFrame
            Error analysis summary
        """
        logger.info("Performing error analysis")
        
        error_data = []
        
        for label in self.class_labels:
            mask = y_true == label
            total = mask.sum()
            errors = (y_true[mask] != y_pred[mask]).sum()
            error_rate = (errors / total * 100) if total > 0 else 0
            
            error_data.append({
                'Class': label,
                'Total': total,
                'Errors': errors,
                'Error Rate (%)': error_rate,
                'Accuracy (%)': 100 - error_rate
            })
        
        error_df = pd.DataFrame(error_data)
        
        # Overall statistics
        total_errors = (y_true != y_pred).sum()
        total_samples = len(y_true)
        
        logger.info(f"\nOverall: {total_errors}/{total_samples} errors ({total_errors/total_samples*100:.1f}%)")
        logger.info(f"\nPer-class error rates:")
        logger.info(error_df.to_string(index=False))
        
        return error_df
    
    def confusion_patterns(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Identify most common misclassification patterns.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        top_n : int
            Number of top patterns to return
            
        Returns
        -------
        pd.DataFrame
            Top confusion patterns
        """
        # Filter to only misclassifications
        misclassified_mask = y_true != y_pred
        
        if misclassified_mask.sum() == 0:
            logger.warning("No misclassifications found!")
            return pd.DataFrame()
        
        y_true_errors = y_true[misclassified_mask]
        y_pred_errors = y_pred[misclassified_mask]
        
        # Count patterns
        patterns = pd.DataFrame({
            'true_label': y_true_errors,
            'predicted_label': y_pred_errors
        })
        
        pattern_counts = patterns.groupby(['true_label', 'predicted_label']).size()
        pattern_counts = pattern_counts.reset_index(name='count')
        pattern_counts = pattern_counts.sort_values('count', ascending=False).head(top_n)
        
        logger.info(f"\nTop {top_n} misclassification patterns:")
        logger.info(pattern_counts.to_string(index=False))
        
        return pattern_counts
    
    def critical_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        critical_classes: List[str] = None
    ) -> Dict[str, int]:
        """
        Identify critical misclassifications.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        critical_classes : List[str], optional
            Classes considered critical (default: ['blocker', 'critical'])
            
        Returns
        -------
        dict
            Critical error counts
        """
        if critical_classes is None:
            critical_classes = ['blocker', 'critical']
        
        # Critical bugs missed (predicted as normal)
        critical_mask = np.isin(y_true, critical_classes)
        missed = (y_true[critical_mask] != y_pred[critical_mask]) & (y_pred[critical_mask] == 'normal')
        critical_missed = missed.sum()
        
        # False alarms (normal predicted as critical)
        normal_mask = y_true == 'normal'
        false_alarms = np.isin(y_pred[normal_mask], critical_classes).sum()
        
        results = {
            'critical_missed': critical_missed,
            'false_alarms': false_alarms,
            'ratio': critical_missed / false_alarms if false_alarms > 0 else np.inf
        }
        
        logger.info(f"\nCritical Errors:")
        logger.info(f"  Critical/Blocker missed (as normal): {critical_missed}")
        logger.info(f"  False alarms (normal as critical): {false_alarms}")
        
        return results