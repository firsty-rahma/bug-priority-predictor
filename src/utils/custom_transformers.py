"""
Custom sklearn transformers.
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
from scipy.sparse import hstack
import pandas as pd

class FeatureCombiner(BaseEstimator, TransformerMixin):
    """Combine TF-IDF text features with categorical features"""
    
    def __init__(self, max_features=1000, min_df=2, max_df=0.8, ngram_range=(1,1)):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.tfidf = None
        self.encoder_component = None
        self.encoder_product = None
        
    def fit(self, X, y=None):
        # TF-IDF for text
        self.tfidf = TfidfVectorizer(
            max_features = self.max_features,
            min_df = self.min_df,
            max_df = self.max_df,
            ngram_range = self.ngram_range
        )
        self.tfidf.fit(X['text_processed'])
        
        # OrdinalEncoder handles unknown categories
        self.encoder_component = OrdinalEncoder(
            handle_unknown='use_encoded_value',  # Assign unknown to -1
            unknown_value=-1
        )
        self.encoder_component.fit(X[['component_name']])
        
        self.encoder_product = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        self.encoder_product.fit(X[['product_name']])
        
        return self
    
    def transform(self, X):
        # Transform text
        X_tfidf = self.tfidf.transform(X['text_processed'])
        
        # Transform categorical (handles unknown values)
        component_encoded = self.encoder_component.transform(X[['component_name']])
        product_encoded = self.encoder_product.transform(X[['product_name']])
        text_length = X['text_length'].values.reshape(-1, 1)
        
        # Combine all features
        X_combined = hstack([X_tfidf, component_encoded, product_encoded, text_length])
        
        return X_combined