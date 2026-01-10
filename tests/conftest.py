"""
Pytest configuration and shared fixtures.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_bug_data():
    """Sample bug data for testing."""
    return pd.DataFrame({
        'bug_id': ['1', '2', '3', '4', '5'],
        'short_description': [
            'Firefox crashes on startup',
            'Minor UI typo',
            'Critical security vulnerability',
            'Normal feature request',
            'Application hangs'
        ],
        'long_description': [
            'The browser crashes immediately when launched',
            'Small typo in settings menu',
            'SQL injection vulnerability found',
            'Would like to add dark mode',
            'Application freezes when loading large files'
        ],
        'component_name': ['General', 'UI', 'Security', 'General', 'Performance'],
        'product_name': ['FIREFOX', 'FIREFOX', 'FIREFOX', 'FIREFOX', 'FIREFOX'],
        'severity_category': ['blocker', 'trivial', 'critical', 'normal', 'major']
    })

@pytest.fixture
def preprocessed_bug_data():
    """Preprocessed bug data for testing."""
    return pd.DataFrame({
        'text_processed': [
            'firefox crash startup',
            'minor ui typo',
            'critical security vulnerability',
            'normal feature request',
            'application hang'
        ],
        'component_name': ['General', 'UI', 'Security', 'General', 'Performance'],
        'product_name': ['FIREFOX', 'FIREFOX', 'FIREFOX', 'FIREFOX', 'FIREFOX'],
        'text_length': [3, 3, 3, 3, 2],
        'severity_category': ['blocker', 'trivial', 'critical', 'normal', 'major']
    })

@pytest.fixture
def sample_predictions():
    """Sample prediction results for testing."""
    y_true = np.array(['blocker', 'critical', 'major', 'minor', 'normal', 'trivial'])
    y_pred = np.array(['blocker', 'normal', 'normal', 'minor', 'normal', 'normal'])
    confidence = np.array([0.9, 0.6, 0.5, 0.8, 0.95, 0.4])
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'confidence': confidence
    }


@pytest.fixture
def temp_directory(tmp_path):
    """Temporary directory for testing file operations."""
    return tmp_path

