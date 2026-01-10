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
        data = sample_bug_data.copy()
        
        # Add severity_code column with matching length
        data['severity_code'] = data.index % 6
        
        # Clean
        cleaned = clean_raw_data(data)
        
        # severity_code should be removed
        assert 'severity_code' not in cleaned.columns
        
        # Other columns should remain
        assert 'severity_category' in cleaned.columns
        assert len(cleaned) == len(data)
    
    def test_clean_raw_data_without_severity_code(self, sample_bug_data):
        """Test cleaning data that doesn't have severity_code."""
        data = sample_bug_data.copy()
        
        # Make sure no severity_code exists
        if 'severity_code' in data.columns:
            data = data.drop('severity_code', axis=1)
        
        # Should not fail
        cleaned = clean_raw_data(data)
        
        assert len(cleaned) == len(data)
        assert 'severity_category' in cleaned.columns
    
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
        incomplete_data = pd.DataFrame({
            'text_processed': ['test'],
            'severity_category': ['normal']
        })
        
        filepath = temp_directory / 'incomplete.csv'
        incomplete_data.to_csv(filepath, index=False)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Missing required columns"):
            load_preprocessed_data(filepath)
    
    def test_load_raw_data_with_path_object(self, sample_bug_data, temp_directory):
        """Test that loading works with pathlib.Path objects."""
        filepath = temp_directory / 'test_data.csv'
        
        save_data(sample_bug_data, filepath)
        
        # Load using Path object
        loaded = load_raw_data(filepath)
        
        assert len(loaded) == len(sample_bug_data)