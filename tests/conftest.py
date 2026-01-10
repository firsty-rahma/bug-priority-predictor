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
    """Sample bug data for testing (enough data for stratified split)."""
    # Create enough samples for each class (at least 2 per class)
    data = []
    
    # Blocker bugs (3 samples)
    for i in range(3):
        data.append({
            'bug_id': f'blocker_{i}',
            'short_description': 'Firefox crashes on startup',
            'long_description': 'The browser crashes immediately when launched causing data loss',
            'component_name': 'General',
            'product_name': 'FIREFOX',
            'severity_category': 'blocker'
        })
    
    # Critical bugs (3 samples)
    for i in range(3):
        data.append({
            'bug_id': f'critical_{i}',
            'short_description': 'Critical security vulnerability found',
            'long_description': 'SQL injection vulnerability allows unauthorized access to database',
            'component_name': 'Security',
            'product_name': 'FIREFOX',
            'severity_category': 'critical'
        })
    
    # Major bugs (3 samples)
    for i in range(3):
        data.append({
            'bug_id': f'major_{i}',
            'short_description': 'Application hangs on large files',
            'long_description': 'Application freezes when loading large files over 100MB',
            'component_name': 'Performance',
            'product_name': 'FIREFOX',
            'severity_category': 'major'
        })
    
    # Minor bugs (3 samples)
    for i in range(3):
        data.append({
            'bug_id': f'minor_{i}',
            'short_description': 'Minor display issue in settings',
            'long_description': 'Text alignment is slightly off in the settings panel',
            'component_name': 'UI',
            'product_name': 'FIREFOX',
            'severity_category': 'minor'
        })
    
    # Normal bugs (6 samples - more common)
    for i in range(6):
        data.append({
            'bug_id': f'normal_{i}',
            'short_description': 'Feature request for dark mode',
            'long_description': 'Would like to add dark mode theme to the application',
            'component_name': 'General',
            'product_name': 'FIREFOX',
            'severity_category': 'normal'
        })
    
    # Trivial bugs (3 samples)
    for i in range(3):
        data.append({
            'bug_id': f'trivial_{i}',
            'short_description': 'Small typo in menu',
            'long_description': 'Minor typo in the help menu needs correction',
            'component_name': 'UI',
            'product_name': 'FIREFOX',
            'severity_category': 'trivial'
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def preprocessed_bug_data():
    """Preprocessed bug data for testing (matches sample_bug_data structure)."""
    data = []
    
    # Blocker (3)
    for i in range(3):
        data.append({
            'text_processed': 'crash startup browser crash immediately launch cause data loss',
            'component_name': 'General',
            'product_name': 'FIREFOX',
            'text_length': 10,
            'severity_category': 'blocker'
        })
    
    # Critical (3)
    for i in range(3):
        data.append({
            'text_processed': 'security vulnerability sql injection vulnerability allow unauthorized access database',
            'component_name': 'Security',
            'product_name': 'FIREFOX',
            'text_length': 10,
            'severity_category': 'critical'
        })
    
    # Major (3)
    for i in range(3):
        data.append({
            'text_processed': 'application hang large application freeze load large',
            'component_name': 'Performance',
            'product_name': 'FIREFOX',
            'text_length': 8,
            'severity_category': 'major'
        })
    
    # Minor (3)
    for i in range(3):
        data.append({
            'text_processed': 'minor display issue setting text alignment slightly setting panel',
            'component_name': 'UI',
            'product_name': 'FIREFOX',
            'text_length': 10,
            'severity_category': 'minor'
        })
    
    # Normal (6)
    for i in range(6):
        data.append({
            'text_processed': 'feature request dark mode like add dark mode theme application',
            'component_name': 'General',
            'product_name': 'FIREFOX',
            'text_length': 11,
            'severity_category': 'normal'
        })
    
    # Trivial (3)
    for i in range(3):
        data.append({
            'text_processed': 'small typo menu minor typo help menu need correction',
            'component_name': 'UI',
            'product_name': 'FIREFOX',
            'text_length': 10,
            'severity_category': 'trivial'
        })
    
    return pd.DataFrame(data)


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