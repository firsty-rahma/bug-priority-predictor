.PHONY: help install install-dev test test-fast test-cov format lint clean train predict

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make install-dev  - Install dev dependencies"
	@echo "  make test         - Run all tests with coverage"
	@echo "  make test-fast    - Run tests without coverage"
	@echo "  make test-cov     - Run tests and open coverage report"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Lint code with flake8"
	@echo "  make clean        - Remove cache and temp files"
	@echo "  make train        - Train the model"
	@echo "  make predict      - Run predictions"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html
	@echo "Opening coverage report..."
	@python -m webbrowser htmlcov/index.html

test-integration:
	pytest tests/test_integration.py -v

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

lint:
	flake8 src/ scripts/ tests/
	pylint src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .coverage htmlcov/ .mypy_cache/

train:
	python scripts/01_data_exploration.py
	python scripts/02_text_preprocessing.py
	python scripts/03_modeling.py
	python scripts/04_hyperparameter_tuning.py

predict:
	python scripts/predict.py