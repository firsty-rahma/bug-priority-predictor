@echo off
echo ========================================
echo Installing Bug Severity Classification
echo ========================================

echo.
echo Creating virtual environment...
python -m venv venv

echo.
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

echo.
echo Downloading NLTK data...
python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"

echo.
echo Running tests...
pytest tests/ -v

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo To activate environment: venv\Scripts\activate
echo To run tests: pytest tests/ -v
pause