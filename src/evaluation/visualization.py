"""
Visualization utilities for model evaluation.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class EvaluationVisualizer:
    """Create visualizations for model evaluation."""
    
    def __init__(self, class_labels: List[str]):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        class_labels : List[str]
            List of class labels in order
        """
        self.class_labels = class_labels
        sns.set_style('whitegrid')
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        title : str
            Plot title
        save_path : Path, optional
            Path to save figure
            
        Returns
        -------
        matplotlib.Figure
            Figure object
        """
        cm = confusion_matrix(y_true, y_pred, labels=self.class_labels)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_labels,
            yticklabels=self.class_labels,
            ax=ax
        )
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Saved: {save_path}")
        
        return fig
    
    def plot_error_analysis(
        self,
        error_df: pd.DataFrame,
        confusion_patterns: pd.DataFrame,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create comprehensive error analysis visualization.
        
        Parameters
        ----------
        error_df : pd.DataFrame
            Error analysis data
        confusion_patterns : pd.DataFrame
            Confusion patterns data
        save_path : Path, optional
            Path to save figure
            
        Returns
        -------
        matplotlib.Figure
            Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Error rate by class
        axes[0].bar(
            error_df['Class'],
            error_df['Error Rate (%)'],
            color='coral',
            alpha=0.7
        )
        axes[0].set_xlabel('Severity Class', fontsize=12)
        axes[0].set_ylabel('Error Rate (%)', fontsize=12)
        axes[0].set_title('Error Rate by Class', fontweight='bold', fontsize=14)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Top confusion patterns
        if len(confusion_patterns) > 0:
            top_n = min(8, len(confusion_patterns))
            patterns = confusion_patterns.head(top_n)
            labels = [f"{row['true_label']} → {row['predicted_label']}" 
                     for _, row in patterns.iterrows()]
            
            axes[1].barh(range(len(labels)), patterns['count'].values, color='steelblue', alpha=0.7)
            axes[1].set_yticks(range(len(labels)))
            axes[1].set_yticklabels(labels, fontsize=10)
            axes[1].set_xlabel('Number of Errors', fontsize=12)
            axes[1].set_title('Top Misclassification Patterns', fontweight='bold', fontsize=14)
            axes[1].invert_yaxis()
            axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Saved: {save_path}")
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 30,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Parameters
        ----------
        feature_names : List[str]
            Feature names
        importances : np.ndarray
            Feature importance values
        top_n : int
            Number of top features to show
        save_path : Path, optional
            Path to save figure
            
        Returns
        -------
        matplotlib.Figure
            Figure object
        """
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values, fontsize=9)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Most Important Features', fontweight='bold', fontsize=14)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Saved: {save_path}")
        
        return fig