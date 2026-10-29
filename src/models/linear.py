"""
Linear text classifier for bug severity.

Primary model: logistic regression over near-raw TF-IDF text.

Why this replaces Random Forest + SMOTE:
- The old preprocessing removed negations ("not", "no", "nor") and digits.
- SMOTE interpolated ordinal-encoded categories into fractional values.
- Raw text + linear models scored ~0.35 macro F1 vs ~0.32 for the RF
  baseline across five random splits.
"""

from typing import Sequence, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from utils.config import RAW_TOKEN_PATTERN

TEXT_COLUMN = "text_for_model"
DEFAULT_TEXT_COLUMNS = ("short_description", "long_description")

# Parameter that GridSearchCV uses to address C for each strategy.
MULTICLASS_C_PARAM = {
    "ovr": "classifier__estimator__C",
    "multinomial": "classifier__C",
}


def add_model_text(
    df: pd.DataFrame,
    short_col: str = "short_description",
    long_col: str = "long_description",
    out_col: str = TEXT_COLUMN,
) -> pd.DataFrame:
    """Return a copy of df with a combined near-raw text column."""
    result = df.copy()
    short = result[short_col].fillna("").astype(str)
    long = result[long_col].fillna("").astype(str)
    result[out_col] = (
        (short + " " + long).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    return result


def build_text_vectorizer(
    max_features: int = 20000,
    min_df: int = 2,
    max_df: float = 0.8,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    token_pattern: str = RAW_TOKEN_PATTERN,
) -> TfidfVectorizer:
    """Word-level TF-IDF configured for bug-report text."""
    return TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        lowercase=True,
        token_pattern=token_pattern,
    )


def build_char_vectorizer(
    max_features: int = 20000,
    min_df: int = 3,
    ngram_range: Tuple[int, int] = (3, 5),
) -> TfidfVectorizer:
    """Character n-grams. Useful for typos and identifiers."""
    return TfidfVectorizer(
        analyzer="char_wb",
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
        sublinear_tf=True,
        lowercase=True,
    )


def build_classifier(
    multiclass: str = "ovr",
    C: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
):
    """Build the logistic-regression head for a multiclass strategy."""
    logistic = LogisticRegression(
        C=C,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )
    if multiclass == "ovr":
        return OneVsRestClassifier(logistic, n_jobs=1)
    if multiclass == "multinomial":
        return logistic
    raise ValueError(f"Unknown multiclass strategy: {multiclass!r}")


def build_linear_pipeline(
    C: float = 1.0,
    max_iter: int = 2000,
    max_features: int = 20000,
    min_df: int = 2,
    max_df: float = 0.8,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    use_char_ngrams: bool = False,
    multiclass: str = "ovr",
    text_column: str = TEXT_COLUMN,
    random_state: int = 42,
) -> Pipeline:
    """Build the linear text pipeline.

    The estimator expects a DataFrame containing ``text_column``.
    Use :func:`add_model_text` to create it.

    Parameters
    ----------
    multiclass : {"ovr", "multinomial"}
        "ovr" trains one binary classifier per class; "multinomial" trains
        a single softmax model. Both are kept as candidates because the
        measured difference between them is within split noise.
    """
    blocks = [
        (
            "word",
            build_text_vectorizer(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                ngram_range=ngram_range,
                sublinear_tf=sublinear_tf,
            ),
            text_column,
        )
    ]
    if use_char_ngrams:
        blocks.append(("char", build_char_vectorizer(), text_column))

    features = ColumnTransformer(
        blocks,
        remainder="drop",
        sparse_threshold=1.0,
    )

    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                build_classifier(
                    multiclass=multiclass,
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def linear_param_grid(
    multiclass: str = "ovr",
    max_features: Sequence[int] = (10000, 20000),
    ngram_ranges: Sequence[Tuple[int, int]] = ((1, 1), (1, 2)),
    cs: Sequence[float] = (0.3, 1.0, 3.0),
) -> dict:
    """Compact grid for the linear model, keyed for the chosen strategy."""
    if multiclass not in MULTICLASS_C_PARAM:
        raise ValueError(f"Unknown multiclass strategy: {multiclass!r}")
    return {
        "features__word__max_features": list(max_features),
        "features__word__ngram_range": list(ngram_ranges),
        MULTICLASS_C_PARAM[multiclass]: list(cs),
    }
