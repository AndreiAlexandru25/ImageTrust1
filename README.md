# ImageTrust

**AI-Generated Image Detection with Calibrated Uncertainty and Multi-Backbone Fusion**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

ImageTrust is a forensic application for detecting AI-generated and digitally manipulated images. It uses a multi-backbone embedding fusion approach (ResNet-50 + EfficientNet-B0 + ViT-B/16) with XGBoost and MLP meta-classifiers, temperature-scaled calibration, and conformal prediction for uncertainty quantification.

Developed as a Master's thesis project at the intersection of computer vision, image forensics, and machine learning.

---

## Download & Run (Windows)

**No Python installation required.**

1. Go to [Releases](https://github.com/AndreiAlexandru25/ImageTrust/releases)
2. Download **both** files: `ImageTrust-v1.0.1-win64.7z.001` and `ImageTrust-v1.0.1-win64.7z.002`
3. Place both files in the same folder
4. Extract using [7-Zip](https://www.7-zip.org/) (right-click `.001` file -> 7-Zip -> Extract Here)
5. Run `ImageTrust.exe`

On first launch, the application downloads pre-trained HuggingFace models (~2 GB). Subsequent launches are instant.

### System Requirements

- Windows 10/11 (64-bit)
- 8 GB RAM minimum (16 GB recommended)
- NVIDIA GPU with CUDA support (optional, improves speed)
- Internet connection for first launch (model download)

---

## Results

Evaluated on 604,589 images across 4 compression variants (original, WhatsApp, Instagram, screenshot). Three random seeds, bootstrap 95% confidence intervals.

### Main Comparison

| Method | Accuracy | F1 | ROC-AUC | ECE |
|--------|----------|-----|---------|-----|
| LogReg (ResNet-50 emb.) | 0.870 | 0.839 | 0.939 [0.938, 0.940] | -- |
| XGBoost (ResNet-50 emb.) | 0.885 | 0.854 | 0.956 [0.955, 0.957] | 0.022 |
| XGBoost (EfficientNet-B0 emb.) | 0.881 | 0.848 | 0.953 [0.952, 0.954] | 0.024 |
| XGBoost (ViT-B/16 emb.) | 0.880 | 0.846 | 0.950 [0.949, 0.951] | 0.027 |
| **XGBoost (3-backbone fusion)** | **0.886** | **0.858** | **0.959** | **0.016** |
| **MLP (3-backbone fusion)** | **0.890** | **0.865** | **0.963** | 0.036 |

Multi-backbone fusion improves AUC by +0.3--1.3% over single-backbone baselines. MLP achieves the highest AUC (0.963) while XGBoost has the best calibration (ECE=0.016).

### Degradation Robustness

| Condition | XGBoost AUC | MLP AUC | Drop |
|-----------|-------------|---------|------|
| Original (clean) | 0.961 | 0.964 | -- |
| WhatsApp compression | 0.958 | 0.961 | -0.003 |
| Instagram pipeline | 0.961 | 0.964 | -0.000 |
| Screenshot capture | 0.960 | 0.963 | -0.001 |

The system is robust to social media compression with <0.3% AUC drop.

### Cross-Generator Evaluation (33 Generators)

Evaluated on 33 unseen generators from the GenImage artifacts dataset (5,000 images per generator). The model was NOT trained on these generators -- this tests zero-shot generalization.

**Detected well (>50% TPR):** StarGAN (69%), Denoising DiffGAN (66%), Palette (62%), StyleGAN3 (42%)

**Not detected (<5% TPR):** CycleGAN, DDPM, ProGAN, BigGAN, GLIDE, Latent Diffusion, VQ-Diffusion

This is expected behavior -- the meta-classifier learns embedding-space patterns from its training generators and does not generalize to all architectures. Cross-generator generalization remains an open research problem.

### Statistical Significance

McNemar test: chi2=20.76, p<0.001 (MLP significantly better on predictions).
DeLong test: dAUC=-0.003, p=1.0 (not significant on AUC -- both models are competitive).

### Efficiency

| Component | Time |
|-----------|------|
| ResNet-50 embedding | 4.3 ms/image |
| EfficientNet-B0 embedding | 5.5 ms/image |
| ViT-B/16 embedding | 4.2 ms/image |
| **Total pipeline** | **14.3 ms/image** |

Measured on NVIDIA RTX 5080 (16 GB VRAM), batch size 256, mixed precision.

---

## Architecture

```
Phase 1: Embedding Extraction (GPU)
  Image -> [ResNet-50 (2048-d)] + [EfficientNet-B0 (1280-d)] + [ViT-B/16 (768-d)] + [NIQE (1-d)]
  -> Concatenated 4097-dimensional feature vector per image
  -> Applied to 4 variants: original, WhatsApp, Instagram, screenshot

Phase 2: Meta-Classifier Training
  4097-d features -> XGBoost (GPU-accelerated, 3 seeds)
  4097-d features -> MLP (4097->1024->512->256->1, SWA, mixup, label smoothing)
  -> Temperature scaling calibration
  -> Conformal prediction (LAC/APS/RAPS)

Phase 3: Publication Pipeline
  -> Single-backbone baselines (honest comparison)
  -> Degradation robustness evaluation
  -> Cross-source evaluation
  -> Statistical significance tests
  -> 8 publication figures + 7 LaTeX tables
```

---

## Project Structure

```
imagetrust/
├── src/imagetrust/              # Main source code
│   ├── core/                    # Config, types, exceptions
│   ├── detection/               # ML detection (multi_detector, calibration, models)
│   ├── evaluation/              # Metrics, ablation, cross-generator, degradation
│   ├── explainability/          # Grad-CAM, patch analysis, frequency
│   ├── forensics/               # Forensics engine
│   ├── frontend/                # PySide6 desktop app, Streamlit web UI
│   ├── metadata/                # EXIF, XMP, C2PA provenance
│   ├── reporting/               # PDF/JSON/HTML forensic reports
│   ├── baselines/               # Classical, CNN, ViT baselines
│   └── cli.py                   # Click-based CLI
├── scripts/orchestrator/        # Training & evaluation pipelines
│   ├── run_phase1_pipeline.py   # Phase 1: embedding extraction
│   ├── run_phase2_training.py   # Phase 2: meta-classifier training
│   ├── run_phase3_publication.py # Phase 3: figures, tables, paper
│   └── run_cross_generator_eval.py # Cross-generator evaluation
├── configs/                     # YAML configuration
├── paper/                       # LaTeX paper template
├── tests/                       # Unit + integration tests
├── ImageTrust.spec              # PyInstaller spec for .exe build
└── requirements.txt             # Python dependencies
```

---

## Development Setup

### Prerequisites

- Python 3.10--3.12
- NVIDIA GPU with CUDA 12.x (for training; CPU works for inference)
- 16 GB RAM minimum for training

### Installation

```bash
git clone https://github.com/AndreiAlexandru25/ImageTrust.git
cd imagetrust

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -e ".[dev]"
```

### CLI Commands

```bash
# Analyze a single image
imagetrust analyze photo.jpg

# Start Streamlit web UI
imagetrust ui

# Launch desktop app (PySide6)
imagetrust desktop

# Start API server
imagetrust serve --port 8000
```

### Running Tests

```bash
pytest tests/ -v
pytest tests/unit/ -v --fast     # Skip slow tests
pytest --cov=imagetrust          # With coverage
```

### Building the .exe

```bash
pip install pyinstaller PySide6
python scripts/build_desktop.py
# Output: dist/ImageTrust/ImageTrust.exe
```

---

## Reproducing Results

The full training and evaluation pipeline requires:
- ~30 GB disk for embeddings
- NVIDIA GPU with 16 GB VRAM
- ~2 hours total runtime

```bash
# Phase 1: Extract embeddings from images
python scripts/orchestrator/run_embedding_extraction.py \
    --input_dirs data/train --output_dir data/phase1/embeddings

# Phase 2: Train meta-classifiers (XGBoost + MLP)
python scripts/orchestrator/run_phase2_training.py

# Phase 3: Generate all paper artifacts (figures + tables)
python scripts/orchestrator/run_phase3_publication.py

# Cross-generator evaluation (requires GenImage artifacts dataset)
python scripts/orchestrator/run_cross_generator_eval.py
```

All experiments use fixed seeds (42, 123, 7) for reproducibility. Hardware: RTX 5080, AMD 7800X3D, 32 GB RAM.

---

## Citation

```bibtex
@mastersthesis{imagetrust2026,
  title={ImageTrust: A Heterogeneous Fusion Framework for AI-Generated Image Detection
         with Calibrated Uncertainty and Multi-Label Forensic Verdicts},
  author={Alexandru, Andrei},
  year={2026},
  school={University Name},
  type={Master's Thesis}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
