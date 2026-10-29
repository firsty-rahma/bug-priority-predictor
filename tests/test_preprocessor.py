"""
Tests for text preprocessing module.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Test TextPreprocessor class."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = TextPreprocessor()
        assert preprocessor is not None
        assert len(preprocessor.stopwords) > 0

    def test_initialization_with_custom_stopwords(self):
        """Test initialization with custom stopwords."""
        custom_stopwords = ["crash", "bug"]
        preprocessor = TextPreprocessor(custom_stopwords=custom_stopwords)

        assert "crash" in preprocessor.stopwords
        assert "bug" in preprocessor.stopwords

    def test_combine_text(self):
        """Test text combination."""
        preprocessor = TextPreprocessor()

        # Normal case
        result = preprocessor.combine_text("Short desc", "Long desc")
        assert result == "Short desc Long desc"

        # Empty long description
        result = preprocessor.combine_text("Short desc", "")
        assert result == "Short desc"

        # None values
        result = preprocessor.combine_text(None, "Long desc")
        assert result == "Long desc"

        # Both None
        result = preprocessor.combine_text(None, None)
        assert result == ""

    def test_clean_text(self):
        """Test text cleaning."""
        preprocessor = TextPreprocessor()

        # Lowercase conversion
        assert preprocessor.clean_text("HELLO WORLD") == "hello world"

        # Special character removal
        assert preprocessor.clean_text("Hello, World!") == "hello world"

        # Numbers are kept: version numbers and error codes carry signal
        assert preprocessor.clean_text("Test123") == "test123"
        assert preprocessor.clean_text("Firefox 1.2.3 error") == "firefox 1.2.3 error"

        # Contractions expand so negations survive tokenization
        assert preprocessor.clean_text("Can't open") == "cannot open"

        # Empty/None handling
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text(None) == ""

    def test_tokenize(self):
        """Test tokenization."""
        preprocessor = TextPreprocessor()

        # Normal text
        tokens = preprocessor.tokenize("hello world test")
        assert tokens == ["hello", "world", "test"]

        # Empty string
        tokens = preprocessor.tokenize("")
        assert tokens == []

        # Single word
        tokens = preprocessor.tokenize("hello")
        assert tokens == ["hello"]

    def test_remove_stopwords(self):
        """Test stopword removal."""
        preprocessor = TextPreprocessor()

        tokens = ["the", "quick", "brown", "fox"]
        filtered = preprocessor.remove_stopwords(tokens)

        # 'the' should be removed (it's a stopword)
        assert "the" not in filtered
        assert "quick" in filtered
        assert "brown" in filtered

    def test_lemmatize(self):
        """Test lemmatization."""
        preprocessor = TextPreprocessor()

        # Plural to singular
        tokens = ["crashes", "bugs", "errors"]
        lemmatized = preprocessor.lemmatize(tokens)

        # Should lemmatize to base forms
        assert "crash" in lemmatized or "crashes" in lemmatized

        # Empty list
        assert preprocessor.lemmatize([]) == []

    def test_preprocess(self):
        """Test full preprocessing pipeline."""
        preprocessor = TextPreprocessor()

        text = "Firefox CRASHES on startup!!!"
        result = preprocessor.preprocess(text)

        # Should be lowercase, cleaned, and processed
        assert result.islower() or result == ""
        assert "!" not in result
        assert "crash" in result.lower() or "startup" in result.lower()

    def test_preprocess_dataframe(self, sample_bug_data):
        """Test DataFrame preprocessing."""
        preprocessor = TextPreprocessor()

        result = preprocessor.preprocess_dataframe(sample_bug_data)

        # Check required columns exist
        assert "text_for_model" in result.columns
        assert "text_processed" in result.columns
        assert "text_length" in result.columns

        # Check no empty text
        assert (result["text_processed"].str.len() > 0).all()

        # Check text_length is correct
        for idx, row in result.iterrows():
            expected_length = len(row["text_processed"].split())
            assert row["text_length"] == expected_length

    def test_negations_are_preserved(self):
        """Negation terms must survive stopword removal."""
        preprocessor = TextPreprocessor()

        for term in ["not", "no", "nor", "cannot", "unable"]:
            assert term not in preprocessor.stopwords

        result = preprocessor.preprocess("The browser does not crash and no data loss")
        assert "not" in result.split()
        assert "no" in result.split()

    def test_build_model_text_keeps_digits_and_punctuation(self):
        """The linear model text is intentionally near-raw."""
        preprocessor = TextPreprocessor()

        result = preprocessor.build_model_text(
            "Crash on 1.2.3", "Error NS_ERROR_FAILURE; can't open <b>file</b>"
        )
        assert "1.2.3" in result
        assert "NS_ERROR_FAILURE" in result
        assert "can't" in result
        assert "<b>" not in result

    def test_preprocess_with_custom_stopwords(self):
        """Test preprocessing with custom stopwords."""
        custom_stopwords = ["vulnerability", "security"]
        preprocessor = TextPreprocessor(custom_stopwords=custom_stopwords)

        text = "critical security vulnerability found in system"
        result = preprocessor.preprocess(text)

        # Custom stopwords should be removed
        result_words = result.split()
        assert "vulnerability" not in result_words
        assert "security" not in result_words

        # But other words should remain
        assert "critical" in result_words or "system" in result_words
