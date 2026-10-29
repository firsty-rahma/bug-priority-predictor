# Dependencies Documentation

## Core Dependencies

### Data Science Stack

#### numpy (>=2.3.4)
- **Purpose:** Numerical computing, array operations
- **Used for:** Matrix operations, numerical calculations in ML models
- **Why this version:** Stable release with Python 3.14 support

#### pandas (>=2.3.3)
- **Purpose:** Data manipulation and analysis
- **Used for:** Loading CSV, data cleaning, feature engineering
- **Why this version:** Major performance improvements, better memory usage

#### scipy (>=1.14.0)
- **Purpose:** Scientific computing
- **Used for:** Sparse matrix operations in TF-IDF, statistical functions
- **Why this version:** Installed transitively by scikit-learn; resolved to a Python 3.14-compatible release

---

### Machine Learning

#### scikit-learn (>=1.7.2)
- **Purpose:** Machine learning algorithms
- **Used for:** 
  - RandomForestClassifier, LogisticRegression
  - TfidfVectorizer for text features
  - Train-test split, cross-validation
  - GridSearchCV for hyperparameter tuning
- **Why this version:** Current stable release with Python 3.14 support

#### imbalanced-learn (>=0.14.0)
- **Purpose:** Handle imbalanced datasets
- **Used for:** SMOTE oversampling in the legacy Random Forest baseline
- **Why this version:** Compatible with scikit-learn 1.8+ and Python 3.14

---

### Natural Language Processing

#### nltk (>=3.9.4)
- **Purpose:** Natural language toolkit
- **Used for:**
  - Stopword removal
  - WordNet lemmatization
  - POS tagging (optional; disabled in the final configuration)
- **Required downloads:**
  - `stopwords` - English stopwords list
  - `wordnet` - WordNet lexical database
  - `omw-1.4` - Open Multilingual Wordnet
  - `averaged_perceptron_tagger` - POS tagger
  - `averaged_perceptron_tagger_eng` - POS tagger (newer NLTK)

---

### Visualization

#### matplotlib (>=3.10.7)
- **Purpose:** Plotting library
- **Used for:** Confusion matrices, error analysis plots, feature importance charts
- **Why this version:** Better default styles, improved performance

#### seaborn (>=0.13.2)
- **Purpose:** Statistical visualization
- **Used for:** Enhanced plots with better aesthetics
- **Why this version:** Compatible with matplotlib 3.7+

---

### Utilities

#### tqdm (>=4.67.1)
- **Purpose:** Progress bars
- **Used for:** Show progress during data preprocessing and model training
- **Optional but recommended**

#### joblib (>=1.4.2)
- **Purpose:** Efficient serialization
- **Used for:** Save/load trained models (alternative to pickle)
- **Why this version:** Better compatibility with scikit-learn

---

## Development Dependencies

### Testing

#### pytest (>=9.0.0)
- **Purpose:** Testing framework
- **Used for:** Unit tests, integration tests
- **Why this version:** Current stable release with Python 3.14 support

#### pytest-cov (>=6.0.0)
- **Purpose:** Coverage plugin for pytest
- **Used for:** Measure test coverage
- **Output:** HTML reports showing which code is tested

#### pytest-mock (>=3.11.1)
- **Purpose:** Mocking library
- **Used for:** Mock external dependencies in tests
- **Optional but useful**

#### pytest-xdist (>=3.3.0)
- **Purpose:** Parallel test execution
- **Used for:** Speed up test runs with `-n auto`
- **Optional but faster**

---

### Code Quality

#### black (>=26.3.1)
- **Purpose:** Code formatter
- **Used for:** Automatic code formatting (PEP 8 compliant)
- **Config:** 88 character line length

#### isort (>=8.0.1)
- **Purpose:** Import sorter
- **Used for:** Organize imports alphabetically
- **Compatible with black**

#### flake8 (>=7.3.0)
- **Purpose:** Linter
- **Used for:** Check code style, find potential bugs
- **Config:** Max line length 88 (match black)

#### pylint (>=3.3.4)
- **Purpose:** Advanced linter
- **Used for:** Deep code analysis, enforce best practices
- **Optional but comprehensive**

---

### Type Checking

#### mypy (>=2.0.0)
- **Purpose:** Static type checker
- **Used for:** Verify type hints, catch type errors before runtime
- **Optional but recommended for large projects**

---

### Documentation

#### sphinx (>=7.1.0)
- **Purpose:** Documentation generator
- **Used for:** Generate HTML docs from docstrings
- **Optional but professional**

#### sphinx-rtd-theme (>=1.3.0)
- **Purpose:** ReadTheDocs theme for Sphinx
- **Used for:** Beautiful documentation styling
- **Optional**

---

### Jupyter

#### jupyter (>=1.0.0)
- **Purpose:** Jupyter notebook interface
- **Used for:** Interactive exploration, prototyping
- **Development only**

#### ipykernel (>=6.25.0)
- **Purpose:** Python kernel for Jupyter
- **Used for:** Run Python code in notebooks
- **Development only**

#### notebook (>=7.0.0)
- **Purpose:** Classic Jupyter notebook
- **Used for:** Notebook interface
- **Development only**

---

## Version Constraints Explained

### Why Minimum Versions?
```txt
numpy>=2.3.4
```

- `>=2.3.4` - Minimum version with required features and Python 3.14 support
- No upper bound is set, so major upgrades install automatically; CI must pass before they are considered safe

### Benefits:
- ✅ Get bug fixes and minor improvements automatically
- ✅ Minimum versions are the ones exercised in CI
- ✅ A single dependency set is easier to keep current

---

## Optional Dependencies

### For GPU Support (Advanced)
```bash
# If you want to use GPU acceleration
pip install cupy-cuda11x  # Replace 11x with your CUDA version
```

### For Deployment
```bash
# For API deployment
pip install fastapi uvicorn

# For Docker
# See Dockerfile for containerization
```

---

## Checking Your Installation

### List installed packages
```bash
pip list
```

### Check specific package version
```bash
pip show scikit-learn
```

### Verify all dependencies
```bash
pip check
```

### Export current environment
```bash
pip freeze > requirements-lock.txt
```

This creates a "lock file" with exact versions for reproducibility.

---

## Upgrading Dependencies

### Upgrade all packages
```bash
pip install --upgrade -r requirements.txt
```

### Upgrade specific package
```bash
pip install --upgrade scikit-learn
```

### Upgrade pip itself
```bash
pip install --upgrade pip
```

---

## Uninstalling

### Remove all dependencies
```bash
pip uninstall -r requirements.txt -y
```

### Remove virtual environment
```bash
# Deactivate first
deactivate

# Then delete folder
rm -rf venv/  # Linux/Mac
rmdir /s venv  # Windows
```