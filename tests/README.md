# Testing Guide

## Quick Start

### Install Dependencies
```bash
pip install pytest
```

### Run Tests
```bash
# Basic run
pytest tests/ -v

# Run specific file
pytest tests/test_preprocessor.py -v

# Run specific test
pytest tests/test_preprocessor.py::TestTextPreprocessor::test_clean_text -v
```

## With Coverage (Optional)

### Install Coverage Tool
```bash
pip install pytest-cov
```

### Run with Coverage
```bash
pytest tests/ -v --cov=src --cov-report=html
```

Then open `htmlcov/index.html` in your browser.

## Troubleshooting

### "No module named 'src'"
Add to test file:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

### "NLTK data not found"
Run once:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```