"""
Tests for custom sklearn transformers.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.custom_transformers import FeatureCombiner


class TestFeatureCombiner:
    """Test FeatureCombiner transformer."""
    
    def test_initialization(self):
        """Test FeatureCombiner initialization."""
        combiner = FeatureCombiner(max_features=100, ngram_range=(1, 2))
        
        assert combiner.max_features == 100
        assert combiner.ngram_range == (1, 2)
        assert combiner.tfidf is None  # Not fitted yet
    
    def test_fit_transform(self, preprocessed_bug_data):
        """Test fitting and transforming data."""
        combiner = FeatureCombiner(max_features=50)
        
        X = preprocessed_bug_data[['text_processed', 'component_name', 
                                    'product_name', 'text_length']]
        
        # Fit and transform
        combiner.fit(X)
        X_transformed = combiner.transform(X)
        
        # Check output shape
        assert X_transformed.shape[0] == len(X)
        
        # Should have TF-IDF features + 3 additional features
        # (component, product, text_length)
        assert X_transformed.shape[1] > 3
    
    def test_remove_crash_option(self, preprocessed_bug_data):
        """Test remove_crash option for ablation study."""
        # Create data with 'crash' keyword
        data = preprocessed_bug_data.copy()
        data.loc[0, 'text_processed'] = 'firefox crash startup crash'
        
        X = data[['text_processed', 'component_name', 'product_name', 'text_length']]
        
        # Without removing crash
        combiner1 = FeatureCombiner(max_features=50, remove_crash=False)
        combiner1.fit(X)
        
        # With removing crash
        combiner2 = FeatureCombiner(max_features=50, remove_crash=True)
        combiner2.fit(X)
        
        # The vocabulary should be different
        vocab1 = combiner1.tfidf.get_feature_names_out()
        vocab2 = combiner2.tfidf.get_feature_names_out()
        
        # 'crash' should be in vocab1 if it wasn't filtered
        # but definitely not in vocab2
        assert 'crash' not in vocab2
    
    def test_handle_unknown_categories(self, preprocessed_bug_data):
        """Test handling of unknown categorical values."""
        combiner = FeatureCombiner()
        
        # Fit on training data
        X_train = preprocessed_bug_data[['text_processed', 'component_name', 
                                         'product_name', 'text_length']]
        combiner.fit(X_train)
        
        # Create test data with unknown category
        X_test = pd.DataFrame({
            'text_processed': ['test text'],
            'component_name': ['UnknownComponent'],  # Unknown!
            'product_name': ['FIREFOX'],
            'text_length': [2]
        })
        
        # Should not raise error (handles unknown with -1)
        X_transformed = combiner.transform(X_test)
        assert X_transformed.shape[0] == 1