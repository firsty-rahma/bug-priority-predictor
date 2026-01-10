"""
Tests for data loading module.
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import (
    load_raw_data,
    clean_raw_data,
    save_data,
    load_preprocessed_data
)


class TestDataLoader:
    """Test data loading functions."""
    
    def test_clean_raw_data(self, sample_bug_data):
        """Test cleaning raw data."""
        # Add severity_code column
        data = sample_bug_data.copy()
        data['severity_code'] = [0, 5, 1, 4, 2]
        
        # Clean
        cleaned = clean_raw_data(data)
        
        # severity_code should be removed
        assert 'severity_code' not in cleaned.columns
        
        # Other columns should remain
        assert 'severity_category' in cleaned.columns
        assert len(cleaned) == len(data)
    
    def test_save_and_load_data(self, sample_bug_data, temp_directory):
        """Test saving and loading data."""
        filepath = temp_directory / 'test_data.csv'
        
        # Save
        save_data(sample_bug_data, filepath)
        assert filepath.exists()
        
        # Load
        loaded = load_raw_data(filepath)
        
        # Check data integrity
        assert len(loaded) == len(sample_bug_data)
        assert list(loaded.columns) == list(sample_bug_data.columns)
    
    def test_load_preprocessed_data_validation(self, temp_directory):
        """Test validation when loading preprocessed data."""
        # Create incomplete data
        incomplete_data = pd.DataFrame({
            'text_processed': ['test'],
            'severity_category': ['normal']
            # Missing component_name, product_name
        })
        
        filepath = temp_directory / 'incomplete.csv'
        incomplete_data.to_csv(filepath, index=False)
        
        # Should raise ValueError for missing columns
        with pytest.raises(ValueError, match="Missing required columns"):
            load_preprocessed_data(filepath)