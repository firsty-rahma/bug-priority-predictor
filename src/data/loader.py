"""
Data loading utilities.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def load_raw_data(filepath: Path) -> pd.DataFrame:
    """
    Load raw bug data from CSV
    
    Parameters
    ----------
    filepath: Path
        Path to the CSV file
    
    Returns
    -------
    pd.DataFrame
        Raw bug data
    """
    logger.info(f"Loading raw data from {filepath}")
    data = pd.read_csv(filepath)
    logger.info(f"Loaded {len(data)} rows, {len(data.columns)} columns")
    return data

def load_preprocessed_data(filepath: Path) -> pd.DataFrame:
    """
    Load preprocessed bug data.
    
    Parameters
    ----------
    filepath : Path
        Path to preprocessed CSV file
        
    Returns
    -------
    pd.DataFrame
        Preprocessed data
    """
    logger.info(f"Loading preprocessed data from {filepath}")
    data = pd.read_csv(filepath)

    # Validate required columns
    required_cols = ['text_processed', 'component_name', 'product_name', 
                    'severity_category']
    missing = set(required_cols) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.info(f"Loaded {len(data)} rows")
    return data

def clean_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by removing inconsistent columns.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw bug data
        
    Returns
    -------
    pd.DataFrame
        Cleaned data with severity_code removed
    """
    logger.info("Cleaning raw data")

    # Drop inconsistent severity_code column
    if 'severity_code' in data.columns:
        data = data.drop('severity_code', axis = 1)
        logger.info("Dropped severity_code column (inconsistent mapping)")
    
    return data

def save_data(data: pd.DataFrame, filepath: Path) -> None:
    """
    Save DataFrame to CSV.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to save
    filepath : Path
        Output file path
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(filepath, index=False)
    logger.info(f"Saved {len(data)} rows to {filepath}")