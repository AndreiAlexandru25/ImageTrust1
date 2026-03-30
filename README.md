# ImageTrust

**Multi-Backbone Fusion for AI-Generated Image Detection with Calibrated Uncertainty**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)

ImageTrust is a forensic framework for detecting AI-generated and digitally manipulated images. It uses multi-backbone fusion (ResNet-50 + EfficientNet-B0 + ViT-B/16) with XGBoost and MLP meta-classifiers, temperature-scaled calibration, and conformal prediction for uncertainty quantification.

Developed as a Master's thesis project at **West University of Timisoara**, Faculty of Mathematics and Computer Science.

Submitted to **CISIS 2026** (International Conference on Computational Intelligence in Security for Information Systems).

---

## Table of Contents

1. [Installation from Source](#installation-from-source)
   - [Prerequisites](#prerequisites)
   - [Step 1: Clone the Repository](#step-1-clone-the-repository)
   - [Step 2: Create Virtual Environment](#step-2-create-virtual-environment)
   - [Step 3: Install Dependencies](#step-3-install-dependencies)
   - [Step 4: Verify Installation](#step-4-verify-installation)
3. [Running the Web Application](#running-the-web-application)
   - [Step 5: Start the Backend API](#step-5-start-the-backend-api)
   - [Step 6: Start the Web Frontend](#step-6-start-the-web-frontend)
   - [Step 7: Open in Browser](#step-7-open-in-browser)
4. [Running Tests](#running-tests)
5. [CLI Usage](#cli-usage)
6. [Web Application Features](#web-application-features)
7. [Results](#results)
8. [Architecture](#architecture)
9. [Project Structure](#project-structure)
10. [Reproducing Results](#reproducing-results)
11. [Troubleshooting](#troubleshooting)
12. [Citation](#citation)
13. [License](#license)

---


## Installation from Source

Follow these steps to install and run the full web application (backend + frontend) from source code. Instructions are provided for both **Windows** and **macOS/Linux**.

### Prerequisites

Install the following software **before** proceeding:

| Software | Version | Windows | macOS |
|----------|---------|---------|-------|
| **Python** | 3.10, 3.11, or 3.12 | Download from [python.org](https://www.python.org/downloads/). **Important:** check "Add Python to PATH" during installation. | `brew install python@3.12` (requires [Homebrew](https://brew.sh/)) |
| **Git** | any recent version | Download from [git-scm.com](https://git-scm.com/download/win) | Pre-installed on macOS. Or: `brew install git` |
| **Node.js** | 18 or higher | Download LTS from [nodejs.org](https://nodejs.org/) | `brew install node` |

**Optional:** NVIDIA GPU with CUDA improves inference speed but is **not required**. The application works on CPU.

**Verify prerequisites** (open a terminal / Command Prompt):

```bash
python --version        # Should print Python 3.10.x, 3.11.x, or 3.12.x
git --version           # Should print git version 2.x.x
node --version          # Should print v18.x.x or higher
npm --version           # Should print 9.x.x or higher
```

> **macOS note:** Use `python3` instead of `python` if `python` is not found.

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/CS-Research-Group-UVT/imagetrust.git
cd imagetrust
```

---

### Step 2: Create Virtual Environment

**Windows (Command Prompt or PowerShell):**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux (Terminal):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, your terminal prompt should show `(.venv)` at the beginning.

---

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

This installs all required packages (PyTorch, FastAPI, transformers, etc.). It may take a few minutes.

> **Apple Silicon Macs (M1/M2/M3/M4):** If the installation fails on PyTorch, run this first, then repeat the install:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -e ".[dev]"
> ```

---

> **Note:** From this point on, all commands are identical on Windows, macOS, and Linux.

### Step 4: Verify Installation

```bash
python -c "import imagetrust; print(imagetrust.__version__)"
```

**Expected output:**

```
1.0.1
```

If you see `1.0.1`, the installation was successful. You can also run:

```bash
imagetrust info
```

**Expected output** (details may vary):

```
ImageTrust System Information
   Version: 1.0.1
   Environment: development

PyTorch
   Version: 2.x.x
   CUDA Available: True/False
   ...
```

---

## Running the Web Application

The web application has two parts that run in **two separate terminals**:
- **Backend** (Python/FastAPI) -- serves the AI detection API on port 8000
- **Frontend** (Next.js) -- serves the web interface on port 3000

### Step 5: Start the Backend API

Make sure your virtual environment is activated (you should see `(.venv)` in your prompt).

```bash
imagetrust serve --port 8000
```

**First launch:** The system downloads pre-trained HuggingFace models (~2 GB). This happens only once.

**Expected output** (after model loading):

```
INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     Starting ImageTrust API...
INFO:     Detector loaded successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Keep this terminal open.** The backend must stay running while you use the web application.

**Verify the API is running** -- open a new terminal and run:

```bash
curl http://localhost:8000/health
```

**Expected output:**

```json
{"status":"healthy"}
```

You can also open http://localhost:8000/docs in your browser to see the interactive API documentation (Swagger UI).

---

### Step 6: Start the Web Frontend

Open a **second terminal** (keep the backend running in the first one).

```bash
cd web
npm install
npm run dev
```

`npm install` downloads frontend dependencies (only needed the first time). `npm run dev` starts the development server.

**Expected output:**

```
  - ready started server on 0.0.0.0:3000, url: http://localhost:3000
```

---

### Step 7: Open in Browser

Open your browser and go to:

**http://localhost:3000**

You should see the ImageTrust web interface. To test it:

1. Click "Upload Image" or drag and drop any image (JPEG, PNG)
2. Wait for the analysis to complete (a few seconds)
3. View the verdict: **Real**, **AI-Generated**, or **Uncertain**
4. Explore the detailed results: per-model breakdown, Grad-CAM heatmaps, metadata, forensic report

**To stop the application:** Press `Ctrl+C` in both terminals.

---

## Running Tests

Make sure your virtual environment is activated. Run from the project root directory (not `web/`).

```bash
python -m pytest tests/ -v
```

**Expected output:**

```
========================= test session starts =========================
...
================ 142 passed, 1 skipped, XX warnings in XXs =============
```

All 142 tests should pass. Additional test commands:

```bash
# Unit tests only (faster, no model loading)
python -m pytest tests/unit -v

# Skip slow tests
python -m pytest tests/ -v -m "not slow"

# Tests with coverage report
python -m pytest tests/ --cov=imagetrust --cov-report=term
```

---

## CLI Usage

ImageTrust includes a command-line interface for quick analysis without the web UI:

```bash
# Show system information
imagetrust info

# Analyse a single image
imagetrust analyze path/to/image.jpg

# Analyse all images in a directory
imagetrust batch path/to/folder/

# Start the API server
imagetrust serve --port 8000

# Launch desktop application (requires: pip install PySide6)
imagetrust desktop

# List all available commands
imagetrust --help
```

---

## Web Application Features

- Drag & drop image upload
- Real-time analysis with progress tracking
- Verdict display (Real / AI-Generated / Uncertain) with calibrated confidence
- Per-model breakdown (CNN ensemble, HuggingFace models, signal analysis)
- Grad-CAM heatmaps and patch localisation
- EXIF metadata and C2PA provenance inspection
- Forensic report export (JSON)
- Dark/light theme
- Responsive design

### Web Architecture

```
Browser (localhost:3000)  -->  Next.js Frontend  -->  FastAPI Backend (localhost:8000)
                                   |                        |
                              Zustand store           Multi-tier detection
                              Radix UI + Tailwind     Phase 2 meta-classifiers
                              Framer Motion           HuggingFace models
                                                      Forensics engine
```

In development, Next.js proxies `/api/*` requests to `localhost:8000` automatically.

---

## Results

Evaluated on 604,589 images across 4 compression variants (original, WhatsApp, Instagram, screenshot), with 90,692 test images. Three random seeds, bootstrap 95% confidence intervals.

### Main Comparison

In-domain detection on 90,692 test images. All values in %.

| Method | Acc | Prec | Rec | F1 | AUC | ECE |
|--------|-----|------|-----|-----|-----|-----|
| Wang et al. (2020) | 61.6 | 62.1 | 7.5 | 13.4 | 68.6 | 0.364 |
| Ojha et al. (2023) | 70.3 | 68.5 | 46.5 | 55.4 | 78.7 | 0.206 |
| LogReg + ResNet-50 | 87.0 | 87.8 | 80.2 | 83.9 | 93.9 | 0.020 |
| XGBoost + ResNet-50 | **90.5** | 94.0 | 83.3 | **88.3** | **96.4** | **0.011** |
| XGBoost + EfficientNet-B0 | 90.1 | **94.1** | 82.3 | 87.8 | 96.1 | **0.011** |
| XGBoost + ViT-B/16 | 89.4 | 93.3 | 81.4 | 86.9 | 95.6 | **0.011** |
| **ImageTrust XGBoost (3-backbone)** | 88.7 | 90.5 | 81.9 | 85.9 | 96.0 | 0.016 |
| **ImageTrust MLP (3-backbone)** | 89.1 | 88.4 | **85.2** | 86.8 | 96.3 | 0.039 |

Multi-backbone fusion achieves the highest AUC (96.3%) and recall (85.2%). Single-backbone XGBoost + ResNet-50 has the best accuracy (90.5%) and calibration (ECE=0.011). Both ImageTrust models significantly outperform external baselines: +17.6% AUC and +31.4% F1 over Ojha et al.

### Degradation Robustness

Robustness to social-media compression (AUC %).

| Method | Clean | WhatsApp | Instagram | Screenshot | Avg |
|--------|-------|----------|-----------|------------|-----|
| Wang et al. | 71.7 | 63.6 (-8.1) | 67.5 (-4.2) | 71.6 (-0.1) | 68.6 |
| Ojha et al. | 79.9 | 76.6 (-3.3) | 78.4 (-1.5) | 79.9 (-0.0) | 78.7 |
| LogReg + ResNet-50 | 94.1 | 93.7 (-0.4) | 94.0 (-0.2) | 93.8 (-0.4) | 93.9 |
| XGB + ResNet-50 | 95.7 | 95.5 (-0.3) | 95.6 (-0.1) | 95.6 (-0.2) | 95.6 |
| **IT (XGB)** | **96.1** | 95.8 (-0.3) | **96.1** (-0.1) | 96.0 (-0.1) | **96.0** |
| **IT (MLP)** | **96.4** | **96.1** (-0.3) | **96.4** (-0.0) | **96.3** (-0.1) | **96.3** |

ImageTrust degrades by at most -0.3% AUC under social media compression, compared to -8.1% for Wang et al. and -3.3% for Ojha et al.

### Cross-Generator Evaluation (24 Generators)

Evaluated on 156,550 images from 24 unseen generators (GenImage dataset). The model was NOT trained on these generators -- this tests zero-shot generalisation. Best TPR shown (XGB or MLP).

**Detected well (>40% TPR):** StarGAN (69.3% MLP), Denoising Diffusion GAN (65.9% XGB), Palette (61.9% XGB), StyleGAN3 (42.0% XGB)

**Not detected (<5% TPR):** CycleGAN, DDPM, ProGAN, BigGAN, GLIDE, Latent Diffusion, VQ-Diffusion

**Real image accuracy:** AFHQ 100.0%, Landscape 100.0%, LSUN 99.96%, ImageNet 99.92%, COCO 99.88%. False positives on synthetic-looking datasets: MetFaces (19.0%), SFHQ (3.0%).

This is expected behaviour -- the meta-classifier learns embedding-space patterns from its training generators and does not generalise to all architectures. Cross-generator generalisation remains an open research problem.

### Statistical Significance

McNemar test: chi2 = 20.76, p < 0.001 (MLP significantly better on individual predictions).
DeLong test: p = 1.0 (not significant on AUC -- both models are competitive).

### Efficiency

| Component | Time |
|-----------|------|
| ResNet-50 embedding | 4.3 ms/image |
| EfficientNet-B0 embedding | 5.5 ms/image |
| ViT-B/16 embedding | 4.2 ms/image |
| **Total pipeline** | **14.3 ms/image (70 img/s)** |

Measured on NVIDIA RTX 5080 (16 GB VRAM), batch size 256, mixed precision.

---

## Architecture

```
Phase 1: Embedding Extraction (GPU)
  Image -> [ResNet-50 (2048-d)] + [EfficientNet-B0 (1280-d)] + [ViT-B/16 (768-d)]
  -> Concatenated 4,096-dimensional feature vector per image
  -> Applied to 4 variants: original, WhatsApp, Instagram, screenshot

Phase 2: Meta-Classifier Training
  4,096-d features -> XGBoost (GPU-accelerated, 3 seeds)
  4,096-d features -> MLP (4096->1024->512->256->1, SWA, mixup, label smoothing)
  -> Temperature scaling calibration
  -> Conformal prediction (LAC/APS/RAPS)

Phase 3: Forensic System
  Tier 1: Phase 2 XGBoost + LAC conformal (threshold=0.7652, coverage=95.19%)
  Tier 2: Calibrated CNN ensemble (3 backbones, temperature-scaled)
  Tier 3: 4 HuggingFace models + 5 signal analysers (FFT, noise, texture, edge, colour)
  + Copy-move detection, Grad-CAM, EXIF/XMP, C2PA provenance, screenshot detection
```

---

## Project Structure

```
ImageTrust/
├── src/imagetrust/              # Main source code
│   ├── api/                     # FastAPI REST API (routes, middleware)
│   ├── core/                    # Config, types, exceptions
│   ├── detection/               # ML detection (multi_detector, calibration, models)
│   ├── evaluation/              # Metrics, ablation, cross-generator, degradation
│   ├── explainability/          # Grad-CAM, patch analysis, frequency
│   ├── forensics/               # 12+ forensic plugins (ELA, noise, JPEG, etc.)
│   ├── frontend/                # PySide6 desktop application
│   ├── metadata/                # EXIF, XMP, C2PA provenance
│   ├── reporting/               # PDF/JSON/HTML forensic reports
│   ├── baselines/               # Classical, CNN, ViT baselines
│   └── cli.py                   # Click-based CLI
├── web/                         # Next.js web frontend
│   ├── src/app/                 # Pages (home, analysis)
│   ├── src/components/          # UI components (upload, verdict, heatmaps)
│   ├── src/stores/              # Zustand state management
│   └── src/lib/                 # API client, types, utils
├── scripts/orchestrator/        # Training & evaluation pipelines
├── configs/                     # YAML configuration
├── tests/                       # Unit + integration tests (142 tests)
├── .gitattributes               # Git LFS tracking configuration
├── pyproject.toml               # Python package configuration
└── requirements.txt             # Python dependencies
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

# Phase 3: Generate all paper artefacts (figures + tables)
python scripts/orchestrator/run_phase3_publication.py

# Cross-generator evaluation (requires GenImage dataset)
python scripts/orchestrator/run_cross_generator_eval.py
```

All experiments use fixed seeds (42, 123, 7) for reproducibility. Hardware: RTX 5080, AMD Ryzen 7 7800X3D, 32 GB RAM.

---

## Troubleshooting

### `python` command not found (Windows)

Re-install Python from [python.org](https://www.python.org/downloads/) and make sure to check **"Add Python to PATH"** during installation. Alternatively, try `python3` or `py` instead of `python`.

### `python` command not found (macOS)

Use `python3` instead of `python`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### PyTorch installation fails on Apple Silicon Mac

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

### `imagetrust` command not found

Make sure your virtual environment is activated. You should see `(.venv)` in your terminal prompt. If not:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Backend starts but frontend shows "Failed to fetch" or blank page

Make sure the backend is running on port 8000 before starting the frontend. The frontend expects the API at `http://localhost:8000`.

### `npm install` fails

Make sure Node.js 18+ is installed: `node --version`. If you see an older version, update Node.js from [nodejs.org](https://nodejs.org/).

### Tests fail with "httpx not installed"

Run `pip install -e ".[dev]"` again -- httpx is included in dev dependencies.

### CUDA not detected

The application works fine on CPU. CUDA is optional and only speeds up inference. If you have an NVIDIA GPU, install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) matching your GPU.

---

## Future Work

1. **Continual learning**: Experience replay or elastic weight consolidation to adapt to new generators (e.g., Flux, DALL-E 4) without catastrophic forgetting.
2. **Adversarial robustness**: Adversarial training and input purification to defend against white-box evasion attacks targeting the embedding space.
3. **C2PA-first filtering**: Images with valid C2PA provenance chains bypass ML detection entirely, reducing false positives for authenticated content.
4. **Lightweight deployment**: Knowledge distillation to a single backbone for mobile/edge inference while preserving fusion benefits.
5. **Video support**: Extend frame-level detection with temporal consistency analysis for deepfake video forensics.

---

## Limitations

- No adversarial evaluation performed; the system has not been tested against targeted evasion attacks.
- Zero-shot cross-generator generalisation averages 17--20% TPR across unseen architectures.
- Domain bias towards faces and natural scenes; performance on medical, satellite, or artistic images is untested.
- False positives on synthetic-looking real photographs (SFHQ, MetFaces).
- Performance may degrade for generators released after the training data cutoff.


---

## License

MIT License. See [LICENSE](LICENSE) for details.
