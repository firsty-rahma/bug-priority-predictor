"""
Text preprocessing utilities with POS-aware lemmatization.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import logging
import re
import string
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import nltk

logger = logging.getLogger(__name__)


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
        use_pos_lemmatization: bool = True
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
            'stopwords',
            'wordnet',
            'omw-1.4',
            'averaged_perceptron_tagger'  # For POS tagging
        ]
        
        for package in required_packages:
            try:
                nltk.data.find(f'corpora/{package}' if package != 'averaged_perceptron_tagger' 
                              else f'taggers/{package}')
            except LookupError:
                logger.info(f"Downloading NLTK package: {package}")
                nltk.download(package, quiet=True)
    
    def _build_stopword_list(
        self, 
        custom_stopwords: Optional[List[str]] = None
    ) -> set:
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
        stop_words = set(stopwords.words('english'))
        
        # Clean special characters from standard stopwords
        cleaned_stop_words = {re.sub(r'[^a-z]', '', s) for s in stop_words}
        
        # Add default extended stopwords if none provided
        if custom_stopwords is None:
            custom_stopwords = self._get_default_custom_stopwords()
        
        # Add custom stopwords
        cleaned_stop_words.update(custom_stopwords)
        
        return cleaned_stop_words
    
    @staticmethod
    def _get_default_custom_stopwords() -> set:
        """
        Get default extended stopword list for bug reports.
        
        Returns
        -------
        set
            Extended stopword set
        """
        custom_stopwords = {
            # Platform/tool specific
            'always', 'firefox', 'mozilla', 'gecko', 'bugzilla', 'os', 'com', 'nt',
            'http', 'window', 'windows', 'enus',
            
            # Procedural language
            'reproduce', 'reproduced', 'reproducing', 'reproduces', 
            'reproducible', 'reproducibly',
            'see', 'saw', 'sees', 'seeing', 'seen',
            'step', 'steps', 'stepped', 'stepping',
            'report', 'reported', 'reports', 'reporting', 'reporter',
            'testcases', 'testcase',
            'expect', 'expected', 'expects',
            
            # Generic actions
            'use', 'uses', 'used', 'using',
            'get', 'gets', 'got', 'gotten', 'getting',
            'try', 'tries', 'tried', 'trying',
            'open', 'opens', 'opened', 'opening',
            'close', 'closes', 'closed', 'closing',
            'click', 'clicks', 'clicked', 'clicking',
            'produce', 'produces', 'produced', 'producing',
            'build', 'builds', 'built', 'building',
            
            # Generic nouns
            'line', 'lines',
            'result', 'results', 'resulting', 'resulted',
            'file', 'files', 'filed',
            'page', 'pages', 'paged',
            
            # Temporal
            'new', 'news', 'newly', 'newer', 'newest',
            'today', 'yesterday',
            'current', 'currently',
            'latest', 'recent', 'recently',
        }
        
        # Add single letters (noise from tokenization)
        custom_stopwords.update(string.ascii_lowercase)
        
        return custom_stopwords
    
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
    
    def clean_text(self, text: str) -> str:
        """
        Clean text: lowercase, remove special characters.
        
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
        
        # Lowercase
        text = str(text).lower()
        
        # Remove special characters, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
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
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
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
        return ' '.join(tokens)
    
    def preprocess_dataframe(
        self, 
        df: pd.DataFrame,
        show_progress: bool = True
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
        required_cols = ['short_description', 'long_description']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if show_progress:
            logger.info("Preprocessing DataFrame")
        
        df = df.copy()
        
        # Combine descriptions
        if show_progress:
            logger.info("Combining descriptions...")
        
        df['text_combined'] = df.apply(
            lambda row: self.combine_text(
                row['short_description'], 
                row['long_description']
            ), 
            axis=1
        )
        
        # Preprocess text
        if show_progress:
            logger.info("Preprocessing text...")
            if self.use_pos_lemmatization:
                logger.info("  Using POS-aware lemmatization (may take longer)...")
        
        df['text_processed'] = df['text_combined'].apply(self.preprocess)
        
        # Calculate text length
        df['text_length'] = df['text_processed'].str.split().str.len()
        
        # Remove rows with empty text
        initial_len = len(df)
        df = df[
            (df['text_processed'].notna()) & 
            (df['text_processed'].str.strip() != '')
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
            'total_stopwords': len(self.stopwords),
            'use_pos_lemmatization': self.use_pos_lemmatization,
            'sample_stopwords': sorted(list(self.stopwords))[:20]
        }


# Convenience function for quick preprocessing
def preprocess_text(
    text: str, 
    custom_stopwords: Optional[List[str]] = None,
    use_pos: bool = True
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
        custom_stopwords=custom_stopwords,
        use_pos_lemmatization=use_pos
    )
    return preprocessor.preprocess(text)