"""
Configuration settings for bug severity classification.
"""
from pathlib import Path
import string

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
BEST_MODEL_PATH = MODEL_DIR / "best_model_random_forest_tuned.pkl"

# Severity categories (in order)
SEVERITY_CLASSES = ["blocker", "critical", "major", "minor", "normal", "trivial"]

# Text preprocessing settings
STOPWORD_STRATEGY = 'extended'  # Options: 'minimal', 'extended', 'adaptive'

# Text preprocessing settings
USE_POS_LEMMATIZATION = True  # Set to False for faster preprocessing

# Extended custom stopwords for bug reports
CUSTOM_STOPWORDS = {
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

# Add single letters
CUSTOM_STOPWORDS.update(string.ascii_lowercase)

# Model settings
TFIDF_MAX_FEATURES = 1000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.8
NGRAM_RANGE = (1, 1)  # Based on hyperparameter tuning results

# Training settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Hyperparameter grids
RF_PARAM_GRID = {
    'feature_combiner__ngram_range': [(1, 1), (1, 2)],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, 30],
    'classifier__min_samples_split': [2, 10]
}

LR_PARAM_GRID = {
    'feature_combiner__ngram_range': [(1, 1), (1, 2)],
    'classifier__C': [0.1, 1, 10],
    'classifier__penalty': ['l2']
}