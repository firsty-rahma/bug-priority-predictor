"""
Text preprocessing utilities with POS-aware lemmatization.
"""

import logging
import re
from typing import List, Optional

import nltk
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from utils.config import CUSTOM_STOPWORDS, NEGATION_WORDS

logger = logging.getLogger(__name__)

# Expand contractions before punctuation stripping. Without this, "can't"
# is split into "can" + "t", and "can" is an NLTK stopword, so the
# negation disappears from the processed text entirely.
CONTRACTIONS = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not",
}

_CONTRACTION_RE = re.compile(
    r"\b(?:" + "|".join(map(re.escape, CONTRACTIONS)) + r")\b",
    flags=re.IGNORECASE,
)


class TextPreprocessor:
    """
    Text preprocessing pipeline for bug reports with POS-aware lemmatization.

    Features:
    - Text cleaning (lowercase, special character removal)
    - Custom domain-specific stopword removal
    - POS-aware lemmatization for improved accuracy
    """

    def __init__(
        self,
        custom_stopwords: Optional[List[str]] = None,
        use_pos_lemmatization: bool = True,
    ):
        """
        Initialize preprocessor.

        Parameters
        ----------
        custom_stopwords : List[str], optional
            Additional domain-specific stopwords to remove.
            If None, uses default extended stopword list.
        use_pos_lemmatization : bool, default=True
            Whether to use POS-aware lemmatization (more accurate but slower)
        """
        self.use_pos_lemmatization = use_pos_lemmatization
        self.lemmatizer = WordNetLemmatizer()

        # Download NLTK data if needed
        self._download_nltk_data()

        # Build stopword list
        self.stopwords = self._build_stopword_list(custom_stopwords)

        logger.info(
            f"Initialized TextPreprocessor with {len(self.stopwords)} stopwords "
            f"(POS lemmatization: {use_pos_lemmatization})"
        )

    def _download_nltk_data(self) -> None:
        """Download required NLTK data packages."""
        required_packages = [
            ("corpora", "stopwords"),
            ("corpora", "wordnet"),
            ("corpora", "omw-1.4"),
            ("taggers", "averaged_perceptron_tagger"),
            ("taggers", "averaged_perceptron_tagger_eng"),
        ]

        for folder, package in required_packages:
            try:
                nltk.data.find(f"{folder}/{package}")
            except LookupError:
                logger.info(f"Downloading NLTK package: {package}")
                nltk.download(package, quiet=True)

    def _build_stopword_list(self, custom_stopwords: Optional[List[str]] = None) -> set:
        """
        Build comprehensive stopword list.

        Parameters
        ----------
        custom_stopwords : List[str], optional
            Custom stopwords to add

        Returns
        -------
        set
            Complete stopword set
        """
        # Start with standard English stopwords
        stop_words = set(stopwords.words("english"))

        # Normalize the same way clean_text does so entries match tokens
        cleaned_stop_words = {self.clean_text(word) for word in stop_words}
        cleaned_stop_words = {word for word in cleaned_stop_words if word}

        # Never remove negation or contrast terms
        cleaned_stop_words -= NEGATION_WORDS

        if custom_stopwords is None:
            custom_stopwords = self._get_default_custom_stopwords()

        cleaned_stop_words.update(custom_stopwords)
        cleaned_stop_words -= NEGATION_WORDS

        return cleaned_stop_words

    @staticmethod
    def _get_default_custom_stopwords() -> set:
        """
        Return the reduced, boilerplate-only stopword set.

        Domain terms such as "crash", "file", "window", "open", "close",
        and "click" are intentionally kept: they carry severity signal.
        """
        return set(CUSTOM_STOPWORDS)

    def combine_text(self, short_desc: str, long_desc: str) -> str:
        """
        Combine short and long descriptions.

        Parameters
        ----------
        short_desc : str
            Short description
        long_desc : str
            Long description

        Returns
        -------
        str
            Combined text
        """
        short = str(short_desc) if pd.notna(short_desc) else ""
        long = str(long_desc) if pd.notna(long_desc) else ""
        return f"{short} {long}".strip()

    def build_model_text(self, short_desc: str, long_desc: str) -> str:
        """
        Build the near-raw text consumed by the linear model.

        Deliberately minimal: combine, strip URLs/HTML, collapse whitespace.
        Case, punctuation, digits, and negations are preserved. TF-IDF
        lowercases internally, so lowercasing here would be redundant.

        Parameters
        ----------
        short_desc : str
            Short description
        long_desc : str
            Long description

        Returns
        -------
        str
            Combined text for the model
        """
        combined = self.combine_text(short_desc, long_desc)
        combined = re.sub(r"https?://\S+|www\.\S+", " ", combined)
        combined = re.sub(r"<[^>]+>", " ", combined)
        return re.sub(r"\s+", " ", combined).strip()

    def clean_text(self, text: str) -> str:
        """
        Normalize text before tokenization.

        Keeps digits, version numbers, error codes, and the characters
        that make up technical identifiers. Contractions are expanded so
        negations survive tokenization.

        Parameters
        ----------
        text : str
            Input text

        Returns
        -------
        str
            Cleaned text
        """
        if pd.isna(text) or text == "":
            return ""

        text = str(text)
        text = _CONTRACTION_RE.sub(
            lambda match: CONTRACTIONS[match.group(0).lower()], text
        )
        text = text.lower()

        # Boilerplate only: URLs and HTML tags
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)

        # Keep letters, digits, apostrophes, and intra-word separators
        text = re.sub(r"[^a-z0-9'\s._-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Parameters
        ----------
        text : str
            Input text

        Returns
        -------
        List[str]
            List of tokens
        """
        if not text:
            return []
        return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.

        Parameters
        ----------
        tokens : List[str]
            Input tokens

        Returns
        -------
        List[str]
            Filtered tokens
        """
        return [token for token in tokens if token not in self.stopwords]

    @staticmethod
    def get_wordnet_pos(treebank_tag: str) -> str:
        """
        Convert Treebank POS tag to WordNet POS tag.

        Parameters
        ----------
        treebank_tag : str
            Penn Treebank POS tag (e.g., 'VBD', 'NN', 'JJ')

        Returns
        -------
        str
            WordNet POS tag (ADJ, VERB, NOUN, ADV)

        Examples
        --------
        >>> get_wordnet_pos('VBD')  # Past tense verb
        'v'
        >>> get_wordnet_pos('NN')   # Noun
        'n'
        >>> get_wordnet_pos('JJ')   # Adjective
        'a'
        """
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to noun

    def lemmatize_with_pos(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using POS tags for better accuracy.

        This method is more accurate than default lemmatization because it
        considers the part of speech. For example:
        - "running" (verb) -> "run"
        - "running" (noun) -> "running" (as in "the running of the program")

        Parameters
        ----------
        tokens : List[str]
            Input tokens

        Returns
        -------
        List[str]
            Lemmatized tokens

        Examples
        --------
        >>> lemmatize_with_pos(['crashes', 'running', 'better'])
        ['crash', 'run', 'good']
        """
        if not tokens:
            return []

        # Get POS tags
        pos_tags = pos_tag(tokens)

        # Lemmatize with appropriate POS
        lemmatized = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos))
            for word, pos in pos_tags
        ]

        return lemmatized

    def lemmatize_simple(self, tokens: List[str]) -> List[str]:
        """
        Simple lemmatization without POS tagging (faster but less accurate).

        Parameters
        ----------
        tokens : List[str]
            Input tokens

        Returns
        -------
        List[str]
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens (uses POS-aware or simple based on initialization).

        Parameters
        ----------
        tokens : List[str]
            Input tokens

        Returns
        -------
        List[str]
            Lemmatized tokens
        """
        if self.use_pos_lemmatization:
            return self.lemmatize_with_pos(tokens)
        else:
            return self.lemmatize_simple(tokens)

    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline.

        Pipeline steps:
        1. Clean text (lowercase, remove special chars)
        2. Tokenize
        3. Remove stopwords
        4. Lemmatize (with or without POS tagging)

        Parameters
        ----------
        text : str
            Raw text

        Returns
        -------
        str
            Preprocessed text

        Examples
        --------
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.preprocess("Firefox CRASHES when opening files!!!")
        'crash open file'
        """
        # Clean
        text = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize(text)

        # Remove stopwords
        tokens = self.remove_stopwords(tokens)

        # Lemmatize
        tokens = self.lemmatize(tokens)

        # Rejoin
        return " ".join(tokens)

    def preprocess_dataframe(
        self, df: pd.DataFrame, show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess entire DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with short_description and long_description columns
        show_progress : bool, default=True
            Whether to show progress messages

        Returns
        -------
        pd.DataFrame
            Data with text_processed and text_length columns

        Raises
        ------
        ValueError
            If required columns are missing
        """
        # Validate required columns
        required_cols = ["short_description", "long_description"]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if show_progress:
            logger.info("Preprocessing DataFrame")

        df = df.copy()

        # Combine descriptions
        if show_progress:
            logger.info("Combining descriptions...")

        df["text_combined"] = df.apply(
            lambda row: self.combine_text(
                row["short_description"], row["long_description"]
            ),
            axis=1,
        )

        # Near-raw text for the primary linear model
        df["text_for_model"] = df.apply(
            lambda row: self.build_model_text(
                row["short_description"], row["long_description"]
            ),
            axis=1,
        )

        # Legacy cleaned text for the Random Forest baseline
        if show_progress:
            logger.info("Preprocessing text...")
            if self.use_pos_lemmatization:
                logger.info("  Using POS-aware lemmatization (may take longer)...")

        df["text_processed"] = df["text_for_model"].apply(self.preprocess)

        # Calculate text length
        df["text_length"] = df["text_processed"].str.split().str.len()

        # Remove rows with empty text
        initial_len = len(df)
        df = df[
            (df["text_processed"].notna()) & (df["text_processed"].str.strip() != "")
        ].copy()
        removed = initial_len - len(df)

        if removed > 0:
            logger.warning(f"Removed {removed} rows with empty text")

        if show_progress:
            logger.info(f"Preprocessing complete: {len(df)} rows")
            logger.info(f"Average text length: {df['text_length'].mean():.1f} words")
            logger.info(f"Median text length: {df['text_length'].median():.1f} words")

        return df

    def get_stopword_summary(self) -> dict:
        """
        Get summary of stopword configuration.

        Returns
        -------
        dict
            Stopword statistics
        """
        return {
            "total_stopwords": len(self.stopwords),
            "use_pos_lemmatization": self.use_pos_lemmatization,
            "sample_stopwords": sorted(list(self.stopwords))[:20],
        }


# Convenience function for quick preprocessing
def preprocess_text(
    text: str, custom_stopwords: Optional[List[str]] = None, use_pos: bool = True
) -> str:
    """
    Quick text preprocessing function.

    Parameters
    ----------
    text : str
        Input text
    custom_stopwords : List[str], optional
        Custom stopwords
    use_pos : bool, default=True
        Use POS-aware lemmatization

    Returns
    -------
    str
        Preprocessed text

    Examples
    --------
    >>> preprocess_text("Firefox crashes on startup!")
    'crash startup'
    """
    preprocessor = TextPreprocessor(
        custom_stopwords=custom_stopwords, use_pos_lemmatization=use_pos
    )
    return preprocessor.preprocess(text)
