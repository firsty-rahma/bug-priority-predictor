"""
Configuration settings for bug severity classification.
"""

import string
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data files
RAW_DATA_PATH = DATA_DIR / "mozilla_bug_report_data.csv"
CLEANED_DATA_PATH = DATA_DIR / "bugs_cleaned.csv"
PREPROCESSED_DATA_PATH = DATA_DIR / "bugs_preprocessed.csv"

# Model files
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
LEGACY_MODEL_PATH = MODEL_DIR / "best_model_random_forest_tuned.pkl"

# Text columns
RAW_TEXT_COLUMN = "text_for_model"
PROCESSED_TEXT_COLUMN = "text_processed"

# Severity categories (in order)
SEVERITY_CLASSES = ["blocker", "critical", "major", "minor", "normal", "trivial"]

# Text preprocessing settings
STOPWORD_STRATEGY = "minimal"  # Boilerplate only; negations are never removed
USE_POS_LEMMATIZATION = False  # POS tagging did not help; keep preprocessing cheap

# Negation and contrast terms must never be treated as stopwords: a
# bag-of-words model otherwise merges "does not crash" with "crashes".
NEGATION_WORDS = {
    "not",
    "no",
    "nor",
    "never",
    "none",
    "nothing",
    "without",
    "cannot",
    "can't",
    "cant",
    "don't",
    "dont",
    "doesn't",
    "doesnt",
    "didn't",
    "didnt",
    "isn't",
    "isnt",
    "wasn't",
    "wasnt",
    "weren't",
    "werent",
    "won't",
    "wont",
    "wouldn't",
    "wouldnt",
    "shouldn't",
    "shouldnt",
    "couldn't",
    "couldnt",
    "unable",
}

# Boilerplate only. Do NOT add bug-domain terms such as file, page, window,
# open, close, click, line, use, get, or new: they carry severity signal.
CUSTOM_STOPWORDS = {
    "bugzilla",
    "http",
    "https",
    "www",
    "com",
    "enus",
    "testcase",
    "testcases",
    "reproduce",
    "reproduced",
    "reproducing",
    "reproduces",
    "reproducible",
    "reproducibly",
}

# Add single letters (noise from tokenization)
CUSTOM_STOPWORDS.update(string.ascii_lowercase)

# Legacy Random Forest baseline (1000 filtered features)
RF_TFIDF_MAX_FEATURES = 1000
RF_TFIDF_MIN_DF = 2
RF_TFIDF_MAX_DF = 0.8
RF_NGRAM_RANGE = (1, 1)

# Primary linear model (near-raw text)
RAW_TFIDF_MAX_FEATURES = 20000
RAW_TFIDF_MIN_DF = 2
RAW_TFIDF_MAX_DF = 0.8
RAW_NGRAM_RANGE = (1, 2)
RAW_SUBLINEAR_TF = True
# Keeps contractions such as "can't" and "don't" intact. The sklearn
# default splits them into "can" + "t", and "can" is an NLTK stopword.
RAW_TOKEN_PATTERN = r"(?u)\b\w[\w']+\b"

# Multiclass strategies compared by the training script
MULTICLASS_STRATEGIES = ["ovr", "multinomial"]

# Training settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
CV_REPEATS = 2  # Repeated CV: a single 5-fold split is too noisy for selection
EVAL_SEEDS = [42, 0, 1, 2, 3]

# Prefer the default regularisation unless a tuned value wins by more than
# this margin. A 0.003 CV difference is well inside fold noise and did not
# replicate on held-out splits.
DEFAULT_C = 1.0
CV_TOLERANCE = 0.005
