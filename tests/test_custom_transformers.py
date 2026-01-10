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
        combiner = FeatureCombiner(
            max_features=50,
            min_df=1,
            max_df=1.0
        )
        
        X = preprocessed_bug_data[['text_processed', 'component_name', 
                                    'product_name', 'text_length']]
        
        # Fit and transform
        combiner.fit(X)
        X_transformed = combiner.transform(X)
        
        # Check output shape
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] > 3
    
    def test_handle_unknown_categories(self, preprocessed_bug_data):
        """Test handling of unknown categorical values."""
        combiner = FeatureCombiner(min_df=1, max_df=1.0)
        
        X_train = preprocessed_bug_data[['text_processed', 'component_name', 
                                         'product_name', 'text_length']]
        combiner.fit(X_train)
        
        X_test = pd.DataFrame({
            'text_processed': ['test text crash vulnerability'],
            'component_name': ['UnknownComponent'],
            'product_name': ['FIREFOX'],
            'text_length': [4]
        })
        
        X_transformed = combiner.transform(X_test)
        assert X_transformed.shape[0] == 1
    
    def test_feature_combination(self, preprocessed_bug_data):
        """Test that TF-IDF and categorical features are properly combined."""
        combiner = FeatureCombiner(max_features=10, min_df=1, max_df=1.0)
        
        X = preprocessed_bug_data[['text_processed', 'component_name', 
                                    'product_name', 'text_length']]
        
        combiner.fit(X)
        X_transformed = combiner.transform(X)
        
        # Should have at least TF-IDF features + 3 categorical/numeric
        # (component_encoded, product_encoded, text_length)
        assert X_transformed.shape[1] >= 3
        
        # Should be sparse matrix
        from scipy.sparse import issparse
        assert issparse(X_transformed)
    
    def test_multiple_fits(self, preprocessed_bug_data):
        """Test that transformer can be fitted multiple times."""
        combiner = FeatureCombiner(max_features=10, min_df=1, max_df=1.0)
        
        X = preprocessed_bug_data[['text_processed', 'component_name', 
                                    'product_name', 'text_length']]
        
        # First fit
        combiner.fit(X)
        vocab_1 = set(combiner.tfidf.get_feature_names_out())
        
        # Second fit (should reset)
        combiner.fit(X)
        vocab_2 = set(combiner.tfidf.get_feature_names_out())
        
        # Vocabularies should be the same (fitted on same data)
        assert vocab_1 == vocab_2