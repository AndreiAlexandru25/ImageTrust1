# 🔍 ImageTrust

**A Forensic Application for Identifying AI-Generated and Digitally Manipulated Images**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

---

## 📋 Overview

ImageTrust is a comprehensive forensic application designed to detect AI-generated and manipulated images with high accuracy and reliability. Built for research and practical applications, it provides:

- **🎯 AI Detection**: Multi-model ensemble for robust AI image detection
- **📊 Calibrated Confidence**: Reliable probability scores with temperature scaling
- **🔬 Explainability**: Grad-CAM heatmaps and patch-level analysis
- **📋 Metadata Analysis**: EXIF, XMP, and C2PA provenance validation
- **📄 Forensic Reports**: Professional PDF/HTML/JSON reports
- **🌐 REST API**: FastAPI backend for integration
- **💻 Web UI**: Modern Streamlit interface
- **🖥️ Desktop App**: Native Windows application with drag-and-drop
- **📦 Windows .exe**: Standalone executable (no Python required)
- **🧪 Baseline Framework**: Classical, CNN, and ViT baselines with calibration
- **📈 Reproducibility**: Complete experiment reproduction pipeline

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/imagetrust.git
cd imagetrust

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

### Basic Usage

```bash
# Analyze a single image
imagetrust analyze photo.jpg

# Start the web UI
imagetrust ui

# Start the API server
imagetrust serve --port 8000

# Launch desktop application
imagetrust desktop
```

### Python API

```python
from imagetrust.detection import AIDetector

# Create detector
detector = AIDetector(model="ensemble")

# Detect AI-generated content
result = detector.detect("image.jpg")
print(f"AI Probability: {result['ai_probability']:.1%}")
print(f"Verdict: {result['verdict']}")

# Full analysis with explainability
from PIL import Image
image = Image.open("image.jpg")
analysis = detector.analyze(image, include_explainability=True)
print(analysis.get_summary())
```

---

## 🏗️ Architecture

```
imagetrust/
├── src/imagetrust/
│   ├── api/            # FastAPI REST API
│   ├── baselines/      # Baseline detectors (classical, CNN, ViT)
│   │   ├── classical.py
│   │   ├── cnn.py
│   │   ├── vit.py
│   │   ├── calibration.py    # Probability calibration
│   │   └── uncertainty.py    # Selective prediction
│   ├── cli.py          # Command-line interface
│   ├── core/           # Configuration, types, exceptions
│   ├── data/           # Dataset management
│   │   └── splits.py   # Train/val/test splits
│   ├── desktop/        # PySide6 desktop application
│   │   └── app.py
│   ├── detection/      # AI detection models
│   │   ├── models/     # CNN, ViT, Ensemble
│   │   └── detector.py
│   ├── evaluation/     # Benchmarking & testing
│   │   ├── ablation.py       # Ablation study
│   │   ├── cross_generator.py
│   │   └── degradation.py
│   ├── explainability/ # Grad-CAM, patches, frequency
│   ├── frontend/       # Streamlit web UI
│   ├── metadata/       # EXIF, XMP, C2PA
│   ├── reporting/      # PDF, HTML, JSON reports
│   └── utils/          # Helpers, logging, image utils
├── assets/             # Icons and branding
├── configs/            # Configuration files
├── data/               # Datasets and splits
├── docker/             # Containerization
├── docs/               # Documentation
├── outputs/            # Experiment outputs
├── scripts/            # Evaluation and build scripts
└── tests/              # Unit tests
```

---

## 📊 Performance Comparison

The dissertation requires a comparison against existing methods. A full template and instructions are provided here:

- `docs/performance_comparison.md`

It includes baseline methods, metrics, and how to reproduce results using:
```
python scripts/run_evaluation.py --dataset data/eval
```

---

## 🎯 Key Features

### 1. Multi-Model Detection

- **CNN Backbones**: ConvNeXt, EfficientNet, ResNet
- **Vision Transformers**: ViT, Swin, DeiT
- **Ensemble**: Weighted combination for robustness

### 2. Calibrated Probabilities

- Temperature scaling for reliable confidence
- Bounded to realistic range (80-95%)
- Expected Calibration Error (ECE) metrics

### 3. Explainability

- **Grad-CAM/Grad-CAM++**: Attention heatmaps
- **Patch Analysis**: Region-level scores
- **Frequency Analysis**: FFT-based patterns

### 4. Metadata & Provenance

- **EXIF**: Camera, timestamp, GPS
- **XMP**: Edit history, creator tools
- **C2PA**: Cryptographic provenance

### 5. Robustness Testing

- Cross-generator evaluation
- Degradation robustness (JPEG, blur, noise)
- MLE-STAR inspired ablation study

### 6. Baseline Framework

Compare against reproducible baselines:

```python
from imagetrust.baselines import get_baseline, list_baselines

# Available: classical, cnn, vit, imagetrust
baseline = get_baseline("cnn", backbone="efficientnet_b0")
baseline.fit(train_images, train_labels, val_images, val_labels)

# Predict with calibration
result = baseline.predict_proba(image)
print(f"AI probability: {result.ai_probability:.2%}")
```

### 7. Probability Calibration

Ensure reliable confidence estimates:

```python
from imagetrust.baselines import calibrate_baseline

calibrator, result = calibrate_baseline(
    baseline, val_images, val_labels,
    method="temperature"  # or "platt", "isotonic"
)

print(f"ECE before: {result.ece_before:.4f}")
print(f"ECE after: {result.ece_after:.4f}")
```

### 8. Desktop Application

Native Qt application for Windows:

- Drag-and-drop image analysis
- Dark theme modern interface
- Real-time detection results
- Export results to JSON
- No terminal required

```bash
# Run from command line
imagetrust desktop

# Build standalone .exe
python scripts/build_desktop.py
```

---

## 📊 Evaluation

### Cross-Generator Testing

Tested against multiple AI generators:
- Midjourney
- DALL-E 3
- Stable Diffusion XL
- Adobe Firefly
- Ideogram

### Degradation Robustness

Tested under various conditions:
- JPEG compression (50-95 quality)
- Gaussian blur (σ = 0.5-2.0)
- Resize (25%-75%)
- Gaussian noise (1%-5%)

---

## 🖥️ CLI Commands

```bash
# Analyze single image
imagetrust analyze photo.jpg --output result.json

# Batch analysis
imagetrust batch ./images/ --output results.json

# Evaluate on dataset
imagetrust evaluate --dataset ./testset/ --model ensemble

# Start API server
imagetrust serve --port 8000 --reload

# Launch web UI
imagetrust ui --port 8501

# Launch desktop application
imagetrust desktop

# System info
imagetrust info
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/analyze` | POST | Analyze image |
| `/analyze/batch` | POST | Batch analysis |
| `/model` | GET | Model info |

### Example Request

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@photo.jpg"
```

---

## 📚 Documentation

- [API Reference](docs/API.md) - REST API and Python SDK
- [Deployment Guide](docs/DEPLOYMENT.md) - Windows .exe, Docker, production
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions
- [Thesis Appendix](docs/THESIS_APPENDIX.md) - Hyperparameters, dataset details, full results
- [Architecture](docs/architecture.md) - System design
- [Threat Model](docs/threat_model.md) - Security considerations
- [User Study Protocol](docs/user_study_protocol.md) - Evaluation methodology
- [OpenAPI Docs](http://localhost:8000/docs) - Interactive API explorer

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit -v

# Run integration tests
pytest tests/integration -m integration

# Run with coverage
pytest --cov=imagetrust --cov-report=html

# Skip slow tests
pytest -m "not slow"
```

---

## 🔧 Development

### Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/imagetrust
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

---

## 🐳 Docker

```bash
# Build image
docker build -t imagetrust -f docker/Dockerfile .

# Run container
docker run -p 8000:8000 imagetrust

# With docker-compose
docker-compose -f docker/docker-compose.yml up
```

---

## 📦 Windows Executable

Build a standalone Windows application (no Python required):

```bash
# Install build dependencies
pip install -e ".[desktop]"
pip install pyinstaller>=6.0.0

# Build .exe (folder distribution)
python scripts/build_desktop.py

# Build single-file .exe
python scripts/build_desktop.py --onefile

# Output: dist/ImageTrust/ImageTrust.exe
```

See [Deployment Guide](docs/DEPLOYMENT.md) for distribution checklist and code signing.

---

## 🔬 Reproducibility

Reproduce all thesis experiments with a single command:

```bash
# Full pipeline (data → baselines → evaluation → figures)
python scripts/reproduce_all.py --data-root ./data --output-dir ./outputs

# Dry run (show what would be executed)
python scripts/reproduce_all.py --dry-run

# Run specific stages
python scripts/reproduce_all.py --stage baselines
python scripts/reproduce_all.py --stage calibration
python scripts/reproduce_all.py --stage ablation
```

### Individual Scripts

```bash
# Baseline comparison
python scripts/run_baselines.py --dataset data/test --baseline all --train

# Calibration analysis
python scripts/run_calibration.py --splits-dir data/splits

# Ablation study
python scripts/run_ablation.py --splits-dir data/splits --generate-tables

# Cross-generator evaluation
python scripts/run_cross_generator.py --eval-dir data/cross_generator

# Generate LaTeX figures
python scripts/generate_figures.py --results outputs/baselines/{timestamp}

# Generate LaTeX tables
python scripts/generate_tables.py --results outputs/baselines/{timestamp}

# Statistical significance tests
python scripts/statistical_tests.py --results outputs/baselines/{timestamp}
```

---

## 📝 Citation

If you use ImageTrust in your research, please cite:

```bibtex
@thesis{imagetrust2024,
  title={ImageTrust: A Forensic Application for Identifying AI-Generated and Digitally Manipulated Images},
  author={Your Name},
  year={2024},
  school={Your University},
  type={Master's Thesis}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
- [C2PA](https://c2pa.org/) - Content Authenticity Initiative
- MLE-STAR paper for ablation study methodology

---

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [yourusername](https://github.com/yourusername)

---

**Made with ❤️ for Master's Thesis**
