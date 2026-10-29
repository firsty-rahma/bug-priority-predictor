"""
Tests for the linear text model.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.pipeline import Pipeline

from models.linear import (
    TEXT_COLUMN,
    add_model_text,
    build_classifier,
    build_linear_pipeline,
    build_text_vectorizer,
    linear_param_grid,
)


@pytest.fixture
def text_frame():
    """Small, separable text dataset for pipeline smoke tests."""
    rows = []
    for label, text in [
        ("blocker", "browser crashes on startup data loss"),
        ("blocker", "cannot start firefox crash"),
        ("critical", "critical security issue data leak"),
        ("critical", "severe crash when opening page"),
        ("normal", "please add dark mode option"),
        ("normal", "typo in about dialog"),
    ]:
        for _ in range(4):
            rows.append({TEXT_COLUMN: text, "severity": label})
    return pd.DataFrame(rows)


class TestAddModelText:
    """Test near-raw text construction."""

    def test_combines_and_normalizes(self):
        frame = pd.DataFrame(
            {
                "short_description": ["Crash on 1.2.3"],
                "long_description": ["Error   NS_ERROR_FAILURE\ncan't open"],
            }
        )
        result = add_model_text(frame)

        assert TEXT_COLUMN in result.columns
        assert result[TEXT_COLUMN].iloc[0] == (
            "Crash on 1.2.3 Error NS_ERROR_FAILURE can't open"
        )

    def test_handles_missing_values(self):
        frame = pd.DataFrame(
            {"short_description": [None], "long_description": ["details"]}
        )
        result = add_model_text(frame)

        assert result[TEXT_COLUMN].iloc[0] == "details"


class TestVectorizer:
    """Test TF-IDF tokenization choices."""

    def test_keeps_contractions(self):
        vectorizer = build_text_vectorizer(min_df=1)
        matrix = vectorizer.fit_transform(["can't open", "cannot open"])

        vocabulary = set(vectorizer.get_feature_names_out())
        assert "can't" in vocabulary
        assert "cannot" in vocabulary
        assert matrix.shape[0] == 2

    def test_default_does_not_keep_stopwords_out(self):
        # No stopword list is supplied: "not" must remain a feature.
        vectorizer = build_text_vectorizer(min_df=1)
        vectorizer.fit(["does not crash", "crashes"])

        assert "not" in set(vectorizer.get_feature_names_out())


class TestPipeline:
    """Test pipeline construction and fitting."""

    @pytest.mark.parametrize("multiclass", ["ovr", "multinomial"])
    def test_fit_and_predict(self, text_frame, multiclass):
        pipeline = build_linear_pipeline(
            min_df=1, max_features=200, multiclass=multiclass
        )

        assert isinstance(pipeline, Pipeline)
        pipeline.fit(text_frame[[TEXT_COLUMN]], text_frame["severity"])
        predictions = pipeline.predict(text_frame[[TEXT_COLUMN]])

        assert len(predictions) == len(text_frame)
        assert set(predictions).issubset(set(text_frame["severity"]))

    def test_classifier_types(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        assert isinstance(build_classifier("ovr"), OneVsRestClassifier)
        assert isinstance(build_classifier("multinomial"), LogisticRegression)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            build_classifier("softmax")


class TestParamGrid:
    """Test grid keys match the chosen strategy."""

    def test_ovr_grid_key(self):
        grid = linear_param_grid("ovr")
        assert "classifier__estimator__C" in grid

    def test_multinomial_grid_key(self):
        grid = linear_param_grid("multinomial")
        assert "classifier__C" in grid

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            linear_param_grid("invalid")
