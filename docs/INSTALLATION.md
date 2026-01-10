# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

## Installation Methods

### Method 1: Using pip (Recommended)

#### 1. Clone the repository
```bash
git clone https://github.com/yourusername/bug-severity-classification.git
cd bug-severity-classification
```

#### 2. Create virtual environment (recommended)
```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

#### 3. Install dependencies
```bash
# Production dependencies only
pip install -r requirements.txt

# Or with development dependencies
pip install -r requirements-dev.txt
```

#### 4. Download NLTK data
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger')"
```

#### 5. Verify installation
```bash
python -c "import sklearn, nltk, pandas, numpy; print('All packages installed successfully!')"
```

---

### Method 2: Using Conda

#### 1. Create conda environment
```bash
conda env create -f environment.yml
```

#### 2. Activate environment
```bash
conda activate bug-severity-classification
```

#### 3. Download NLTK data
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger')"
```

---

### Method 3: Install as Package (Advanced)

Install the project as an editable package:
```bash
# Clone repository
git clone https://github.com/yourusername/bug-severity-classification.git
cd bug-severity-classification

# Install in editable mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

This allows you to:
- Import modules anywhere: `from data.preprocessor import TextPreprocessor`
- Use CLI commands: `bug-predict --help`

---

## Troubleshooting

### Issue: "No module named 'sklearn'"

**Solution:**
```bash
pip install scikit-learn
```

### Issue: "NLTK data not found"

**Solution:**
```bash
python -m nltk.downloader stopwords wordnet omw-1.4 averaged_perceptron_tagger
```

### Issue: "Microsoft Visual C++ required" (Windows)

**Solution:** Install Microsoft C++ Build Tools
1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"

### Issue: ImportError with numpy/pandas

**Solution:** Update pip first
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```
### Issue: "ImportError: cannot import name 'SMOTE'"
**Solution:**
```bash
pip install imbalanced-learn
```
---

## Verify Installation

Run this script to verify all dependencies:
```python
# verify_installation.py
import sys

def check_package(package_name, import_name=None):
    """Check if a package can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} - NOT INSTALLED")
        return False

print("Checking dependencies...")
print("=" * 50)

packages = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scikit-learn", "sklearn"),
    ("imbalanced-learn", "imblearn"),
    ("nltk", "nltk"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("pytest", "pytest"),
]

all_installed = all(check_package(pkg, imp) for pkg, imp in packages)

print("=" * 50)
if all_installed:
    print("✓ All packages installed successfully!")
else:
    print("✗ Some packages are missing. Run: pip install -r requirements.txt")

sys.exit(0 if all_installed else 1)
```

**Run verification:**
```bash
python verify_installation.py
```

---

## Quick Start After Installation

### 1. Download sample data (if not included)
```bash
# Place your bugs.csv in data/ directory
mkdir -p data
# Copy your dataset to data/bugs.csv
```

### 2. Run the complete pipeline
```bash
# Data exploration
python scripts/01_data_exploration.py

# Text preprocessing
python scripts/02_text_preprocessing.py

# Model training
python scripts/03_modeling.py

# Hyperparameter tuning
python scripts/04_hyperparameter_tuning.py

# Error analysis
python scripts/05_error_analysis.py
```

### 3. Make predictions
```bash
python scripts/predict.py --short-desc "Firefox crashes on startup"
```

---

## Development Setup

If you plan to contribute or modify the code:
```bash
# Install with dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Format code
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Lint code
flake8 src/ scripts/ tests/
```

---

## System Requirements

### Minimum
- RAM: 4 GB
- Storage: 2 GB free space
- CPU: Any modern processor

### Recommended
- RAM: 8 GB or more
- Storage: 5 GB free space
- CPU: Multi-core processor (for faster training)

---

## Need Help?

- Check [README.md](README.md) for project overview
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- Open an issue on GitHub: [Issues](https://github.com/yourusername/bug-severity-classification/issues)