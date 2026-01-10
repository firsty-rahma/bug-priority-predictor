#!/usr/bin/env python
"""
Data Exploration Script

Explores raw bug data and creates cleaned version.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from utils.config import (
    RAW_DATA_PATH, 
    CLEANED_DATA_PATH,
    FIGURES_DIR
)
from data.loader import load_raw_data, clean_raw_data, save_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def explore_data(data: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw bug data
    """
    logger.info("="*70)
    logger.info("DATA EXPLORATION")
    logger.info("="*70)

    # Basic info
    logger.info(f"\nShape: {data.shape}")
    logger.info(f"Columns: {data.shape[1]}")

    # Missing values
    missing = data.isnull().sum()
    if missing.any():
        logger.info(f"\nMissing values:")
        logger.info(missing[missing > 0])

    # Severity distribution
    logger.info("\nSeverity Category Distribution:")
    logger.info(data['severity_category'].value_counts().sort_index())

    logger.info("\nPercentages:")
    percentages = (data['severity_category'].value_counts(normalize=True) * 100).round(2)
    logger.info(percentages.sort_index())

    # Severity code issue
    if 'severity_code' in data.columns:
        logger.info("\nSeverity Code Distribution:")
        logger.info(data['severity_code'].value_counts().sort_index())

        logger.info("\nCategory-Code Mapping:")
        mapping = data.groupby(['severity_category', 'severity_code']).size()
        logger.info(mapping)
        
        logger.warning("⚠️  Issue: Both 'minor' and 'normal' map to code 2")
        logger.warning("⚠️  Issue: Code 3 is missing")
        logger.info("→ Decision: Will drop severity_code column and use severity_category as the target variable")

def visualize_distributions(data: pd.DataFrame, output_dir: Path) -> None:
    """
    Create visualizations of data distributions.
    
    Parameters
    ----------
    data : pd.DataFrame
        Bug data
    output_dir : Path
        Directory to save figures
    """
    logger.info("\nCreating visualizations...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Severity distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    severity_counts = data['severity_category'].value_counts().sort_index()
    axes[0].bar(range(len(severity_counts)), severity_counts.values, color='steelblue')
    axes[0].set_xticks(range(len(severity_counts)))
    axes[0].set_xticklabels(severity_counts.index, rotation=45)
    axes[0].set_ylabel('Count')
    axes[0].set_title('Severity Category Distribution', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add counts on bars
    for i, v in enumerate(severity_counts.values):
        axes[0].text(i, v + 50, str(v), ha='center', va='bottom')
    
    # Percentage plot
    severity_pct = (data['severity_category'].value_counts(normalize=True) * 100).sort_index()
    axes[1].barh(range(len(severity_pct)), severity_pct.values, color='coral')
    axes[1].set_yticks(range(len(severity_pct)))
    axes[1].set_yticklabels(severity_pct.index)
    axes[1].set_xlabel('Percentage (%)')
    axes[1].set_title('Severity Category Percentage', fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add percentages on bars
    for i, v in enumerate(severity_pct.values):
        axes[1].text(v + 1, i, f'{v:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'severity_distribution.png', dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved: {output_dir / 'severity_distribution.png'}")
    plt.close()

def main():
    """Main execution function."""
    logger.info("Starting data exploration")

    # Load raw data
    data = load_raw_data(RAW_DATA_PATH)

    # Explore
    explore_data(data)

    # Visualize
    visualize_distributions(data, FIGURES_DIR)

    # Clean
    data_cleaned = clean_raw_data(data)

    # Save
    save_data(data_cleaned, CLEANED_DATA_PATH)
    
    logger.info("\n" + "="*70)
    logger.info("✅ DATA EXPLORATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Cleaned data saved to: {CLEANED_DATA_PATH}")

if __name__ == "__main__":
    main()