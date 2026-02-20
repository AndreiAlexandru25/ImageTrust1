# Academic Evaluation Framework for ImageTrust

This document defines the complete evaluation methodology for the Master's thesis on AI-generated image detection.

---

## 1. Baseline Definitions

We define **3 baselines** of increasing complexity to establish a performance spectrum.

### B1: Classical Baseline (LogReg/XGBoost on Handcrafted Features)

| Property | Value |
|----------|-------|
| **Input** | RGB image (any size, resized internally to 512×512) |
| **Output** | P(AI) ∈ [0, 1] |
| **Classifier** | Logistic Regression or XGBoost |
| **Feature Vector** | ~400 dimensions |

**Features Extracted:**

| Feature Group | Count | Description | Rationale |
|---------------|-------|-------------|-----------|
| DCT coefficients | 64 | Low-frequency 2D-DCT on grayscale | GAN spectral fingerprints |
| Noise residuals | 14 | Laplacian filter + cross-channel correlation | Real cameras have characteristic sensor noise |
| JPEG artifacts | 8 | 8×8 block boundary discontinuities | AI images often lack JPEG blocking |
| Color statistics | 108 | Per-channel mean/std/percentiles + histograms | AI color distribution anomalies |
| LBP texture | 256 | Local Binary Pattern histogram (8 neighbors) | Texture micro-patterns differ |
| Edge statistics | 8 | Gradient magnitude + direction histogram | AI images may be over-smoothed |

**Hyperparameters (for reporting):**

```yaml
# Logistic Regression
classifier: logistic_regression
C: 1.0
max_iter: 1000
solver: lbfgs
class_weight: balanced

# XGBoost (alternative)
classifier: xgboost
n_estimators: 100
max_depth: 6
learning_rate: 0.1
subsample: 0.8
```

**Limitations:**
- Cannot capture high-level semantic artifacts
- Sensitive to image preprocessing (resize, compression)
- Features hand-designed, may not generalize to new generators

**Metrics Reported:**
- Accuracy, Balanced Accuracy, F1, ROC-AUC, AP
- Feature extraction time (ms/image)
- Training time
- Feature importance ranking (top-10)

---

### B2: CNN Baseline (ResNet-50 Binary Classifier)

| Property | Value |
|----------|-------|
| **Input** | RGB image, resized to 224×224 |
| **Output** | P(AI) ∈ [0, 1] |
| **Architecture** | ResNet-50 (ImageNet pretrained) + FC head |
| **Parameters** | ~23.5M (backbone) + ~2K (head) |

**Architecture Details:**

```
ResNet-50 backbone (frozen first 2 blocks)
    → AdaptiveAvgPool2d(1, 1)
    → Dropout(0.5)
    → Linear(2048, 512)
    → ReLU
    → Dropout(0.3)
    → Linear(512, 2)
    → Softmax
```

**Training Protocol:**

```yaml
optimizer: AdamW
learning_rate: 1e-4
weight_decay: 1e-4
scheduler: CosineAnnealingLR
epochs: 10
batch_size: 32
early_stopping_patience: 3
augmentation:
  - RandomHorizontalFlip(p=0.5)
  - RandomCrop(224, padding=16)
  - ColorJitter(brightness=0.1, contrast=0.1)
```

**Limitations:**
- Fixed input resolution loses fine details
- May overfit to training generator's artifacts
- Limited receptive field for global inconsistencies

**Metrics Reported:**
- All standard metrics + per-epoch training curves
- Inference time (ms/image, batch size 1 and 32)
- GPU memory usage (if applicable)

---

### B3: ViT Baseline (Vision Transformer)

| Property | Value |
|----------|-------|
| **Input** | RGB image, resized to 224×224 (or 384×384) |
| **Output** | P(AI) ∈ [0, 1] |
| **Architecture** | ViT-B/16 (ImageNet-21k pretrained) |
| **Parameters** | ~86M |

**Architecture Details:**

```
ViT-B/16 backbone
    → [CLS] token embedding (768-dim)
    → LayerNorm
    → Linear(768, 2)
    → Softmax
```

**Training Protocol:**

```yaml
optimizer: AdamW
learning_rate: 1e-5  # Lower for transformers
weight_decay: 0.01
scheduler: LinearWarmup + CosineDecay
warmup_epochs: 1
epochs: 10
batch_size: 16  # Lower due to memory
gradient_accumulation: 2
```

**Limitations:**
- Higher computational cost (GPU recommended)
- Requires more training data for fine-tuning
- Patch-based tokenization may miss pixel-level artifacts

**Metrics Reported:**
- All standard metrics
- Attention map visualizations (optional)
- Throughput (images/second)

---

### ImageTrust (Proposed Method)

| Property | Value |
|----------|-------|
| **Input** | RGB image (native resolution preserved) |
| **Output** | P(AI), confidence interval, explainability maps |
| **Architecture** | Ensemble of pretrained HuggingFace detectors |
| **Calibration** | Temperature scaling on validation set |

**Advantages over baselines:**
- Multi-resolution analysis
- Model ensemble for robustness
- Calibrated probability outputs
- Built-in explainability (Grad-CAM, LIME)
- C2PA provenance verification (if available)

---

## 2. Evaluation Protocol

### 2.1 Dataset Assumptions

Since specific datasets may be private/proprietary, we define the **expected structure**:

```
data/
├── train/
│   ├── real/           # Label 0: Authentic images
│   │   ├── img001.jpg
│   │   └── ...
│   ├── midjourney/     # Label 1: AI-generated
│   ├── dalle3/         # Label 1
│   └── stable_diffusion/  # Label 1
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

**Minimum Requirements:**
- At least 1,000 images per class (real vs AI)
- Multiple generators represented for generalization testing
- Balanced class distribution (or use stratified sampling)

**Recommended Public Datasets:**
- GenImage (partial)
- AI-generated vs Real (Kaggle)
- CIFAKE
- Custom collection from Midjourney/DALL-E APIs

### 2.2 Data Splits

| Split | Ratio | Purpose | Seed |
|-------|-------|---------|------|
| Train | 70% | Model training | 42 |
| Validation | 15% | Hyperparameter tuning, calibration, early stopping | 42 |
| Test | 15% | Final evaluation (held-out) | 42 |

**Stratification:** By class (real/AI) AND by generator (if multiple).

**Create splits with:**
```bash
python scripts/create_splits.py \
    --dataset ./data/raw \
    --output ./data/splits \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42 \
    --stratify-by generator
```

### 2.3 Metrics

**Primary Metrics (always report):**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| **Balanced Accuracy** | (TPR+TNR)/2 | Accounts for class imbalance |
| **F1 Score** | 2·(P·R)/(P+R) | Harmonic mean of precision/recall |
| **ROC-AUC** | Area under ROC curve | Threshold-independent discrimination |
| **Average Precision (AP)** | Area under PR curve | Important for imbalanced data |

**Calibration Metrics:**

| Metric | Description | Target |
|--------|-------------|--------|
| **ECE** | Expected Calibration Error | < 0.05 |
| **MCE** | Maximum Calibration Error | < 0.15 |
| **Brier Score** | Mean squared error of probabilities | < 0.15 |

**Per-Class Metrics:**

| Metric | Description |
|--------|-------------|
| **Precision (Real)** | TP_real / (TP_real + FP_real) |
| **Recall (Real)** | TP_real / (TP_real + FN_real) |
| **Precision (AI)** | TP_ai / (TP_ai + FP_ai) |
| **Recall (AI)** | TP_ai / (TP_ai + FN_ai) |

**Confusion Matrix:**
```
                Predicted
              Real    AI
Actual Real   TN      FP
       AI     FN      TP
```

### 2.4 Evaluation Scenarios

#### A) In-Domain Evaluation
Train and test on same generator distribution.
```bash
python scripts/run_baselines.py \
    --dataset ./data/splits/test \
    --baseline all \
    --output ./outputs/eval_indomain
```

#### B) Cross-Generator Evaluation
Train on subset of generators, test on held-out generators.
```bash
python scripts/run_baselines.py \
    --dataset ./data/splits/test \
    --baseline all \
    --cross-generator \
    --output ./outputs/eval_crossgen
```

Report as **heatmap**: rows = methods, cols = generators.

#### C) Robustness Evaluation
Test degradation tolerance:
```bash
python scripts/run_baselines.py \
    --dataset ./data/splits/test \
    --baseline all \
    --degradation \
    --output ./outputs/eval_robustness
```

**Degradation Parameters:**
| Degradation | Values |
|-------------|--------|
| JPEG compression | Q = 95, 85, 70, 50 |
| Gaussian blur | σ = 0, 0.5, 1.0, 2.0 |
| Resize | factor = 1.0, 0.75, 0.5 |
| Gaussian noise | σ = 0, 0.01, 0.03 |

### 2.5 Ablation Study Plan

Systematically evaluate component contributions:

| Ablation | What's Changed | Expected Impact |
|----------|---------------|-----------------|
| A1: Backbone | EfficientNet vs ConvNeXt vs ViT | Architecture sensitivity |
| A2: Input size | 224 vs 384 vs 512 | Resolution trade-off |
| A3: Ensemble | Single model vs ensemble | Robustness gain |
| A4: Calibration | With vs without temperature scaling | ECE improvement |
| A5: Preprocessing | With vs without augmentation | Generalization |

```bash
python scripts/run_ablation.py \
    --dataset ./data/splits/val \
    --output ./outputs/ablation
```

### 2.6 Statistical Significance

For method comparisons:
- **McNemar's test** for paired binary predictions
- **DeLong's test** for AUC comparison
- Report **95% confidence intervals** via bootstrap (1000 resamples)

```bash
python scripts/statistical_tests.py \
    --predictions ./outputs/eval_indomain/predictions.json \
    --output ./outputs/statistical_tests.json
```

---

## 3. Paper Artifacts

### 3.1 Tables

| Table | Content | File |
|-------|---------|------|
| **Table 1** | Baseline comparison (Acc, F1, AUC, ECE) | `reports/tables/tab1_baselines.tex` |
| **Table 2** | Cross-generator results (AUC per generator) | `reports/tables/tab2_crossgen.tex` |
| **Table 3** | Ablation study results | `reports/tables/tab3_ablation.tex` |
| **Table 4** | Computational cost (time, memory) | `reports/tables/tab4_runtime.tex` |
| **Table 5** | Statistical significance (p-values) | `reports/tables/tab5_significance.tex` |

**Generate with:**
```bash
python scripts/generate_tables.py \
    --results ./outputs/eval_indomain \
    --output ./reports/tables \
    --format latex
```

### 3.2 Figures

| Figure | Content | File |
|--------|---------|------|
| **Fig 1** | System architecture diagram | (manual: draw.io/TikZ) |
| **Fig 2** | ROC curves (all methods) | `reports/figures/fig2_roc_curves.pdf` |
| **Fig 3** | Reliability diagram (calibration) | `reports/figures/fig3_reliability.pdf` |
| **Fig 4** | Cross-generator heatmap | `reports/figures/fig4_crossgen_heatmap.pdf` |
| **Fig 5** | Degradation robustness curves | `reports/figures/fig5_degradation.pdf` |
| **Fig 6** | Confusion matrices (2×2 grid) | `reports/figures/fig6_confusion.pdf` |
| **Fig 7** | Performance bar chart | `reports/figures/fig7_performance_bars.pdf` |
| **Fig 8** | Explainability examples (Grad-CAM) | `reports/figures/fig8_explainability.pdf` |
| **Fig 9** | ECE before/after calibration | `reports/figures/fig9_ece_comparison.pdf` |

**Generate with:**
```bash
# From real results:
python scripts/generate_figures.py \
    --results ./outputs/eval_indomain \
    --output ./reports/figures \
    --format pdf

# Demo figures (synthetic data, for testing layout):
python scripts/generate_figures.py --demo \
    --output ./reports/figures/demo
```

### 3.3 Thesis Section Structure

```
Chapter 4: Methodology
├── 4.1 Problem Formulation
│   └── Binary classification: P(AI|image) → [0,1]
├── 4.2 Baseline Methods
│   ├── 4.2.1 Classical (B1): Feature extraction + LogReg/XGBoost
│   ├── 4.2.2 CNN (B2): ResNet-50 fine-tuning
│   └── 4.2.3 ViT (B3): Vision Transformer
├── 4.3 Proposed Method: ImageTrust
│   ├── 4.3.1 Multi-resolution feature extraction
│   ├── 4.3.2 Ensemble strategy
│   ├── 4.3.3 Probability calibration
│   └── 4.3.4 Explainability module
└── 4.4 Implementation Details
    └── Hyperparameters, training procedure, hardware

Chapter 5: Experiments
├── 5.1 Experimental Setup
│   ├── 5.1.1 Datasets
│   ├── 5.1.2 Evaluation metrics
│   └── 5.1.3 Hardware and software
├── 5.2 In-Domain Evaluation
│   └── Table 1, Figure 2, Figure 6
├── 5.3 Cross-Generator Generalization
│   └── Table 2, Figure 4
├── 5.4 Robustness Analysis
│   └── Figure 5
├── 5.5 Ablation Study
│   └── Table 3
├── 5.6 Calibration Analysis
│   └── Figure 3, Figure 9
└── 5.7 Computational Efficiency
    └── Table 4

Chapter 6: Results and Discussion
├── 6.1 Main Findings
├── 6.2 Comparison with State-of-the-Art
├── 6.3 Limitations
└── 6.4 Future Work
```

---

## 4. Executable Task List

### Phase 1: Setup (Day 1)

```bash
# 1. Verify environment
cd imagetrust
pip install -e ".[dev,baselines]"
python scripts/verify_setup.py

# 2. Prepare dataset structure
mkdir -p data/{train,val,test}/{real,midjourney,dalle3,stable_diffusion}
# Copy your images into appropriate folders

# 3. Create splits (if not already done)
python scripts/create_splits.py \
    --dataset ./data/raw \
    --output ./data/splits \
    --seed 42
```

### Phase 2: Train Baselines (Day 2-3)

```bash
# Train all baselines (CPU-compatible, GPU optional)
python scripts/run_baselines.py \
    --dataset ./data/splits \
    --baseline all \
    --train \
    --epochs 10 \
    --seed 42 \
    --output ./outputs/baselines/run_$(date +%Y%m%d)

# Monitor training (optional, if using GPU)
# tensorboard --logdir ./outputs/baselines/run_*/logs
```

**Expected outputs:**
```
outputs/baselines/run_YYYYMMDD/
├── checkpoints/
│   ├── classical.pth
│   ├── cnn.pth
│   └── vit.pth
├── classical_history.json
├── cnn_history.json
├── vit_history.json
└── reproducibility.json
```

### Phase 3: Evaluate (Day 4)

```bash
# In-domain evaluation
python scripts/run_baselines.py \
    --dataset ./data/splits/test \
    --baseline all \
    --output ./outputs/eval_indomain

# Cross-generator evaluation
python scripts/run_baselines.py \
    --dataset ./data/splits/test \
    --baseline all \
    --cross-generator \
    --output ./outputs/eval_crossgen

# Robustness evaluation
python scripts/run_baselines.py \
    --dataset ./data/splits/test \
    --baseline all \
    --degradation \
    --output ./outputs/eval_robustness
```

**Expected outputs per eval:**
```
outputs/eval_indomain/
├── main_results.json
├── main_results_table.csv
├── main_results_table.tex
├── predictions/
│   ├── classical_predictions.json
│   ├── cnn_predictions.json
│   └── ...
└── reproducibility.json
```

### Phase 4: Ablation Study (Day 5)

```bash
python scripts/run_ablation.py \
    --dataset ./data/splits/val \
    --detector imagetrust \
    --output ./outputs/ablation
```

### Phase 5: Generate Artifacts (Day 6)

```bash
# Generate all figures
python scripts/generate_figures.py \
    --results ./outputs/eval_indomain \
    --output ./reports/figures \
    --format pdf

# Generate all tables
python scripts/generate_tables.py \
    --results ./outputs/eval_indomain \
    --cross-gen ./outputs/eval_crossgen \
    --ablation ./outputs/ablation \
    --output ./reports/tables \
    --format latex

# Statistical tests
python scripts/statistical_tests.py \
    --predictions ./outputs/eval_indomain \
    --output ./reports/tables/statistical_tests.json
```

### Phase 6: Final Reproducibility Check

```bash
# Full reproduction script (runs everything from scratch)
python scripts/reproduce_all.py \
    --dataset ./data/splits \
    --output ./outputs/final_$(date +%Y%m%d) \
    --seed 42

# Verify outputs
ls -la outputs/final_*/
ls -la reports/figures/
ls -la reports/tables/
```

---

## 5. Output Directory Structure

After running all evaluations:

```
outputs/
├── baselines/
│   └── run_20240115/           # Training run
│       ├── checkpoints/
│       ├── *_history.json
│       └── reproducibility.json
├── eval_indomain/              # In-domain test results
│   ├── main_results.json
│   ├── main_results_table.tex
│   └── predictions/
├── eval_crossgen/              # Cross-generator results
│   ├── cross_generator.json
│   └── cross_generator_table.tex
├── eval_robustness/            # Degradation results
│   └── degradation.json
└── ablation/                   # Ablation study
    └── ablation_results.json

reports/
├── figures/
│   ├── fig2_roc_curves.pdf
│   ├── fig3_reliability.pdf
│   ├── fig4_crossgen_heatmap.pdf
│   ├── fig5_degradation.pdf
│   ├── fig6_confusion.pdf
│   ├── fig7_performance_bars.pdf
│   └── fig9_ece_comparison.pdf
└── tables/
    ├── tab1_baselines.tex
    ├── tab2_crossgen.tex
    ├── tab3_ablation.tex
    ├── tab4_runtime.tex
    └── tab5_significance.tex
```

---

## 6. Reproducibility Checklist

Before submitting results, verify:

- [ ] **Seed consistency**: All scripts use `--seed 42`
- [ ] **Split files saved**: `data/splits/` contains JSON split definitions
- [ ] **Checkpoints saved**: All trained models in `outputs/baselines/checkpoints/`
- [ ] **Config logged**: `reproducibility.json` in each output directory
- [ ] **Git commit**: Code version tagged (e.g., `v1.0.0-thesis`)
- [ ] **Requirements pinned**: `pip freeze > requirements-lock.txt`
- [ ] **Hardware documented**: CPU/GPU model, RAM, OS version

---

## 7. Quick Reference: Key Commands

```bash
# Verify setup
python scripts/verify_setup.py

# Quick demo (synthetic data)
python scripts/quick_demo.py

# Full training + evaluation
make train-baselines
make evaluate
make generate-artifacts

# Or all-in-one:
make thesis-full

# Just figures (from existing results)
python scripts/generate_figures.py --results ./outputs/eval_indomain
```

---

## 8. Appendix: Hyperparameter Summary Table

| Baseline | Learning Rate | Batch Size | Epochs | Weight Decay | Optimizer |
|----------|--------------|------------|--------|--------------|-----------|
| B1 (Classical) | N/A | N/A | N/A | C=1.0 | L-BFGS |
| B2 (CNN) | 1e-4 | 32 | 10 | 1e-4 | AdamW |
| B3 (ViT) | 1e-5 | 16 | 10 | 0.01 | AdamW |
| ImageTrust | N/A (pretrained) | 1 | N/A | N/A | N/A |

**Common settings:**
- Random seed: 42
- Early stopping patience: 3 epochs
- Validation metric for stopping: F1 Score
- Device: `cuda` if available, else `cpu`
