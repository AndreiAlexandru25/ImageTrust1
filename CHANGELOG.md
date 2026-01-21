# Changelog

All notable changes to ImageTrust will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

---

## [1.0.0] - 2026-01-20

### Added

#### Core Detection
- Multi-model ensemble AI image detector
- Support for 4 pretrained HuggingFace models:
  - umm-maybe/AI-image-detector
  - Organika/sdxl-detector
  - aiornot/aiornot-detector-v2
  - nyuad/ai-image-detector-2025
- Weighted ensemble voting strategy
- Temperature-scaled probability calibration

#### Baseline Framework
- Classical baseline (hand-crafted features + Logistic Regression)
- CNN baseline (ResNet-50, EfficientNet, ConvNeXt)
- ViT baseline (ViT-B/16, CLIP)
- ImageTrust ensemble baseline
- Unified training and evaluation interface
- Probability calibration (Temperature, Platt, Isotonic scaling)
- Selective prediction with uncertainty estimation

#### Desktop Application
- PySide6/Qt desktop application
- Modern dark theme interface
- Drag-and-drop image analysis
- Real-time detection results
- Export results to JSON
- PyInstaller packaging for Windows .exe

#### Evaluation & Analysis
- Cross-generator evaluation framework
- Degradation robustness testing (JPEG, blur, resize, noise)
- Ablation study framework
- Statistical significance testing (McNemar, DeLong)
- Bootstrap confidence intervals
- Multiple comparison correction (Bonferroni, Holm)

#### Documentation
- Comprehensive API documentation (REST + Python SDK)
- Deployment guide (Windows .exe, Docker, production)
- Troubleshooting guide
- Thesis appendix materials
- Contributing guidelines

#### Scripts & Tools
- `reproduce_all.py` - Master reproducibility pipeline
- `run_baselines.py` - Unified baseline evaluation
- `run_ablation.py` - Ablation study runner
- `run_calibration.py` - Calibration analysis
- `generate_figures.py` - Publication-ready figure generation
- `generate_tables.py` - LaTeX table generation
- `statistical_tests.py` - Significance testing
- `build_desktop.py` - Windows .exe builder

#### Examples
- `01_basic_detection.py` - Single image detection
- `02_batch_processing.py` - Batch processing with progress
- `03_baseline_comparison.py` - Baseline comparison
- `04_calibration_demo.py` - Calibration demonstration

#### CI/CD
- GitHub Actions CI workflow (lint, test, build)
- GitHub Actions release workflow (package, Windows .exe)
- Pre-commit hooks configuration
- Code coverage reporting

#### Configuration
- Centralized hyperparameters configuration (`configs/hyperparameters.yaml`)
- Dataset split management
- Environment-based configuration

### Changed
- Upgraded from Tkinter to PySide6 for desktop application
- Improved calibration with temperature scaling
- Enhanced error handling throughout

### Fixed
- Probability bounds ensuring [0, 1] range
- Memory management for batch processing
- Thread safety in desktop application

---

## [0.2.0] - 2026-01-18

### Added
- Baseline comparison framework
- Dataset split creation and management
- Calibration module with multiple methods
- Cross-generator evaluation
- Degradation robustness testing
- Ablation study infrastructure

### Changed
- Refactored detection module for better extensibility
- Improved logging throughout

---

## [0.1.0] - 2026-01-16

### Added
- Initial project structure
- Core module architecture
- Metadata extraction module (EXIF, XMP)
- C2PA provenance validation
- ML-based AI detection module
- Explainability module (Grad-CAM, patch analysis)
- Evaluation protocol framework
- FastAPI backend
- Streamlit web UI
- Forensic report generator
- CLI interface
- Docker configuration
- Basic test suite

---

## Version History Summary

| Version | Date       | Description                              |
|---------|------------|------------------------------------------|
| 1.0.0   | 2026-01-20 | Thesis release with all features         |
| 0.2.0   | 2026-01-18 | Baseline framework and evaluation        |
| 0.1.0   | 2026-01-16 | Initial development release              |

---

## Upgrade Guide

### From 0.2.x to 1.0.0

1. **Desktop Application**: Now uses PySide6 instead of Tkinter
   ```bash
   pip install -e ".[desktop]"
   ```

2. **Configuration**: Hyperparameters now in `configs/hyperparameters.yaml`

3. **New Commands**:
   ```bash
   imagetrust desktop  # Launch new Qt desktop app
   ```

### From 0.1.x to 0.2.0

1. **Baselines**: New baseline framework available
   ```python
   from imagetrust.baselines import get_baseline
   baseline = get_baseline("cnn")
   ```

2. **Calibration**: New calibration module
   ```python
   from imagetrust.baselines import calibrate_baseline
   ```
