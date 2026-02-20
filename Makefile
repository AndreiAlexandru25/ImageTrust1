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
# Thesis Pipeline (see docs/ACADEMIC_EVALUATION.md for full protocol)
# ==============================================================================

# Quick start: generate demo figures with synthetic data
thesis-demo:
	@echo "Generating demo figures (synthetic data)..."
	python scripts/generate_figures.py --demo --output reports/figures/demo --format pdf
	@echo "Demo figures saved to reports/figures/demo/"

# Full thesis pipeline (requires data in data/splits/)
thesis-full: train-baselines eval-indomain eval-crossgen eval-robustness ablation figures tables stats
	@echo ""
	@echo "============================================"
	@echo "Full thesis pipeline complete!"
	@echo "============================================"
	@echo "Results: outputs/"
	@echo "Figures: reports/figures/"
	@echo "Tables:  reports/tables/"

# Step 1: Create data splits
create-splits:
	@echo "Creating train/val/test splits..."
	python scripts/create_splits.py \
		--dataset ./data/raw \
		--output ./data/splits \
		--train-ratio 0.7 \
		--val-ratio 0.15 \
		--test-ratio 0.15 \
		--seed 42
	@echo "Splits saved to data/splits/"

# Step 2: Train baselines
train-baselines:
	@echo "Training all baselines (CPU, GPU optional)..."
	python scripts/run_baselines.py \
		--dataset ./data/splits \
		--baseline all \
		--train \
		--epochs 10 \
		--seed 42 \
		--output ./outputs/baselines
	@echo "Checkpoints saved to outputs/baselines/checkpoints/"

# Step 3a: In-domain evaluation
eval-indomain:
	@echo "Running in-domain evaluation..."
	python scripts/run_baselines.py \
		--dataset ./data/splits/test \
		--baseline all \
		--output ./outputs/eval_indomain
	@echo "Results in outputs/eval_indomain/"

# Step 3b: Cross-generator evaluation
eval-crossgen:
	@echo "Running cross-generator evaluation..."
	python scripts/run_baselines.py \
		--dataset ./data/splits/test \
		--baseline all \
		--cross-generator \
		--output ./outputs/eval_crossgen
	@echo "Results in outputs/eval_crossgen/"

# Step 3c: Robustness evaluation
eval-robustness:
	@echo "Running degradation robustness tests..."
	python scripts/run_baselines.py \
		--dataset ./data/splits/test \
		--baseline all \
		--degradation \
		--output ./outputs/eval_robustness
	@echo "Results in outputs/eval_robustness/"

# Step 4: Ablation study
ablation:
	@echo "Running ablation study..."
	python scripts/run_ablation.py \
		--dataset ./data/splits/val \
		--output ./outputs/ablation
	@echo "Results in outputs/ablation/"

# Step 5: Calibration analysis
calibration:
	@echo "Running calibration analysis..."
	python scripts/evaluate_calibration.py \
		--dataset ./data/splits/val \
		--output ./outputs/calibration
	@echo "Results in outputs/calibration/"

# Step 6: Generate figures
figures:
	@echo "Generating thesis figures (PDF)..."
	mkdir -p reports/figures
	python scripts/generate_figures.py \
		--results ./outputs/eval_indomain \
		--output ./reports/figures \
		--format pdf
	@echo "Figures saved to reports/figures/"

# Step 7: Generate LaTeX tables
tables:
	@echo "Generating LaTeX tables..."
	mkdir -p reports/tables
	python scripts/generate_tables.py \
		--results ./outputs/eval_indomain \
		--cross-gen ./outputs/eval_crossgen \
		--ablation ./outputs/ablation \
		--output ./reports/tables \
		--format latex
	@echo "Tables saved to reports/tables/"

# Step 8: Statistical significance tests
stats:
	@echo "Running statistical significance tests..."
	mkdir -p reports/tables
	python scripts/statistical_tests.py \
		--predictions ./outputs/eval_indomain \
		--output ./reports/tables/statistical_tests.json
	@echo "Results in reports/tables/statistical_tests.json"

# Legacy aliases for compatibility
baselines: train-baselines eval-indomain
cross-generator: eval-crossgen
degradation: eval-robustness

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
