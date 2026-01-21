# ==============================================================================
# ImageTrust Makefile
# ==============================================================================
# Common commands for development, testing, and deployment

.PHONY: help install install-dev clean lint format test test-cov docs serve run docker-build docker-run

# Default target
help:
	@echo "ImageTrust - Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install development dependencies"
	@echo "  make setup         Full development setup"
	@echo "  make verify        Verify installation is correct"
	@echo ""
	@echo "Development:"
	@echo "  make run           Run the API server"
	@echo "  make ui            Run the Streamlit web UI"
	@echo "  make desktop       Run the desktop application"
	@echo "  make lint          Run linters (ruff, mypy)"
	@echo "  make format        Format code (black, isort)"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo ""
	@echo "Demo:"
	@echo "  make demo          Run quick demo of all features"
	@echo "  make examples      Run example scripts"
	@echo ""
	@echo "Thesis:"
	@echo "  make thesis        Run full thesis pipeline"
	@echo "  make baselines     Train and evaluate baselines"
	@echo "  make ablation      Run ablation study"
	@echo "  make figures       Generate thesis figures"
	@echo "  make tables        Generate LaTeX tables"
	@echo "  make stats         Run statistical tests"
	@echo ""
	@echo "Build:"
	@echo "  make build         Build Python package"
	@echo "  make build-exe     Build Windows executable"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean         Clean build artifacts"
	@echo "  make clean-all     Clean everything including venv"

# ==============================================================================
# Setup
# ==============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

install-all:
	pip install -e ".[all]"
	pre-commit install

setup: install-dev
	@echo "Creating necessary directories..."
	mkdir -p models data/raw data/processed data/evaluation/cross_generator
	mkdir -p data/evaluation/degradation data/evaluation/wild_set
	mkdir -p logs reports outputs/paper/figures outputs/paper/tables
	@echo "Setup complete!"

verify:
	python scripts/verify_setup.py

# ==============================================================================
# Development
# ==============================================================================

run:
	uvicorn imagetrust.api.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	uvicorn imagetrust.api.main:app --host 0.0.0.0 --port 8000 --workers 4

ui:
	streamlit run src/imagetrust/frontend/app.py

desktop:
	python -m imagetrust.desktop.app

lint:
	@echo "Running ruff..."
	ruff check src/imagetrust tests
	@echo "Running mypy..."
	mypy src/imagetrust --ignore-missing-imports

format:
	@echo "Running black..."
	black src/imagetrust tests scripts examples
	@echo "Running isort..."
	isort src/imagetrust tests scripts examples

format-check:
	black --check src/imagetrust tests
	isort --check-only src/imagetrust tests

# ==============================================================================
# Demo
# ==============================================================================

demo:
	python scripts/quick_demo.py

examples:
	@echo "Running example scripts..."
	python examples/01_basic_detection.py --demo
	python examples/04_calibration_demo.py

# ==============================================================================
# Testing
# ==============================================================================

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/imagetrust --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -m "not slow"

test-integration:
	pytest tests/ -v -m "integration"

# ==============================================================================
# Documentation
# ==============================================================================

docs:
	mkdocs build

docs-serve:
	mkdocs serve

# ==============================================================================
# Docker
# ==============================================================================

docker-build:
	docker build -t imagetrust:latest -f docker/Dockerfile .

docker-run:
	docker-compose -f docker/docker-compose.yml up

docker-down:
	docker-compose -f docker/docker-compose.yml down

# ==============================================================================
# Maintenance
# ==============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf venv/
	rm -rf .venv/

# ==============================================================================
# Model & Data Management
# ==============================================================================

download-models:
	python scripts/download_models.py

prepare-data:
	python scripts/prepare_dataset.py

evaluate:
	python scripts/run_evaluation.py

# ==============================================================================
# Release
# ==============================================================================

build:
	python -m build

publish-test:
	python -m twine upload --repository testpypi dist/*

publish:
	python -m twine upload dist/*

# ==============================================================================
# Thesis Pipeline
# ==============================================================================

thesis: baselines calibration ablation figures tables stats
	@echo "Full thesis pipeline complete!"
	@echo "Results in: outputs/paper/"

baselines:
	@echo "Training and evaluating baselines..."
	python scripts/run_baselines.py --dataset data/test --baseline all --train
	@echo "Baseline evaluation complete!"

calibration:
	@echo "Running calibration analysis..."
	python scripts/run_calibration.py --splits-dir data/splits
	@echo "Calibration analysis complete!"

ablation:
	@echo "Running ablation study..."
	python scripts/run_ablation.py --splits-dir data/splits --generate-tables
	@echo "Ablation study complete!"

figures:
	@echo "Generating thesis figures..."
	python scripts/generate_figures.py --output outputs/paper/figures
	@echo "Figures saved to outputs/paper/figures/"

tables:
	@echo "Generating LaTeX tables..."
	python scripts/generate_tables.py --output outputs/paper/tables
	@echo "Tables saved to outputs/paper/tables/"

stats:
	@echo "Running statistical significance tests..."
	python scripts/statistical_tests.py --output outputs/paper/stats
	@echo "Statistical results saved to outputs/paper/stats/"

cross-generator:
	@echo "Running cross-generator evaluation..."
	python scripts/run_cross_generator.py --eval-dir data/evaluation/cross_generator
	@echo "Cross-generator evaluation complete!"

degradation:
	@echo "Running degradation robustness tests..."
	python scripts/run_degradation.py --eval-dir data/evaluation/degradation
	@echo "Degradation tests complete!"

# ==============================================================================
# Build Executables
# ==============================================================================

build-exe:
	@echo "Building Windows executable..."
	python scripts/build_desktop.py
	@echo "Executable built: dist/ImageTrust/ImageTrust.exe"

build-exe-onefile:
	@echo "Building single-file Windows executable..."
	python scripts/build_desktop.py --onefile
	@echo "Executable built: dist/ImageTrust.exe"

# ==============================================================================
# Full Reproducibility
# ==============================================================================

reproduce-all:
	@echo "Running full reproducibility pipeline..."
	python scripts/reproduce_all.py --data-root data --output-dir outputs
	@echo "Full pipeline complete!"

reproduce-dry:
	python scripts/reproduce_all.py --dry-run
