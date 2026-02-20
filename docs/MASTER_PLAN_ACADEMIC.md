# ImageTrust - Academic Master Plan
## Master's Thesis - International Conference B Level (IEEE WIFS / ACM IH&MMSec)

**Date:** February 2026
**Status:** ConvNeXt in training (Epoch 14/25)

---

## Executive Summary

| Phase | Description | Estimated Time | Status |
|-------|-------------|----------------|--------|
| 1 | Setup + Custom Model Training | ~150h | ✅ 85% Done |
| 2 | Complete Academic Evaluation | 4-6h | ⬜ Next |
| 3 | Complete Ablation Study | 2-3h | ⬜ |
| 4 | Calibration + Uncertainty | 1-2h | ⬜ |
| 5 | Screenshot/Recapture Detection | 8-12h | ⬜ NEW |
| 6 | Social Media Recompression | 8-12h | ⬜ NEW |
| 7 | Cross-Generator Evaluation | 2-3h | ⬜ |
| 8 | Degradation Robustness | 2-3h | ⬜ |
| 9 | LaTeX Tables + PDF Figures | 2-4h | ⬜ |
| 10 | Professional UI (PySide6) | 16-24h | ⬜ |
| 11 | Build .exe + Testing | 4-8h | ⬜ |
| 12 | Documentation + Final README | 2-4h | ⬜ |

**Total remaining:** ~50-80h of work after ConvNeXt

---

## PART I: ACADEMIC FOUNDATION (Mandatory for Publication)

### 1.1 Custom Trained Models (✅ In Progress)

```
┌─────────────────────────────────────────────────────────────────┐
│                    BACKBONE MODELS TRAINED                       │
├─────────────────┬──────────┬────────┬────────┬─────────────────┤
│ Model           │ Accuracy │ AUC    │ Recall │ Status          │
├─────────────────┼──────────┼────────┼────────┼─────────────────┤
│ ResNet-50       │ 72.55%   │ 84.97% │ 47.07% │ ✅ Done         │
│ EfficientNetV2-M│ 72.95%   │ 85.44% │ 47.82% │ ✅ Done         │
│ ConvNeXt-Base   │ ???      │ ???    │ ???    │ 🔄 Epoch 14/25  │
└─────────────────┴──────────┴────────┴────────┴─────────────────┘
```

**Critical Observation:** Low recall (~47%) - requires threshold tuning or focal loss.

### 1.2 Baseline Comparison (MANDATORY)

As per the thesis requirements, a comparison with **at least 3 baselines** is needed:

| Baseline | Method | Implementation | Status |
|----------|--------|----------------|--------|
| **Baseline 1** | Classical ML | XGBoost on forensic features (JPEG artifacts, noise, frequency) | ⬜ TODO |
| **Baseline 2** | CNN Single | ResNet-50 fine-tuned (custom trained) | ✅ Done |
| **Baseline 3** | Modern/ViT | HuggingFace pretrained detectors (ensemble of 4 models) | ✅ Done |
| **ImageTrust** | Ensemble + Signal Analysis | Custom trained + HF models + forensics | ✅ Done |

### 1.3 Metrics Required (Paper-Ready)

For each method, the following must be reported:

```
Primary Metrics (Table 1):
├── Accuracy
├── Balanced Accuracy  ← ADDED in metrics.py
├── Precision
├── Recall
├── F1-Score
├── ROC-AUC
├── Average Precision (AP)
└── ECE (Expected Calibration Error)

Secondary Metrics:
├── Specificity
├── NPV (Negative Predictive Value)
├── MCC (Matthews Correlation Coefficient)
└── Cohen's Kappa
```

---

## PART II: ABLATION STUDY (MANDATORY)

### 2.1 Component Ablation

Test the impact of each component:

```
┌──────────────────────────────────────────────────────────────────┐
│                      ABLATION EXPERIMENTS                         │
├──────────────────────────────────┬───────────────────────────────┤
│ Experiment                       │ What It Measures              │
├──────────────────────────────────┼───────────────────────────────┤
│ Full System (Baseline)           │ Complete performance          │
│ - Without Model 1 (Deepfake)     │ Contribution of model 1      │
│ - Without Model 2 (AI-Detector)  │ Contribution of model 2      │
│ - Without Model 3 (AIorNot)      │ Contribution of model 3      │
│ - Without Model 4 (NYUAD)        │ Contribution of model 4      │
│ - Without Signal Analysis        │ Value of signal analysis      │
│ - Without Calibration            │ Impact of calibration         │
│ - Single Best Model Only         │ Benefit of ensemble           │
├──────────────────────────────────┼───────────────────────────────┤
│ Ensemble Strategy Comparison     │                               │
├──────────────────────────────────┼───────────────────────────────┤
│ Average Voting                   │ Simple strategy               │
│ Weighted Voting                  │ With per-model weights        │
│ Majority Voting                  │ Majority vote                 │
│ Max Probability                  │ Most confident                │
│ Median                           │ Robust to outliers            │
├──────────────────────────────────┼───────────────────────────────┤
│ Calibration Methods              │                               │
├──────────────────────────────────┼───────────────────────────────┤
│ No Calibration                   │ Raw probabilities             │
│ Temperature Scaling              │ Single parameter T            │
│ Platt Scaling                    │ Logistic regression           │
│ Isotonic Regression              │ Non-parametric                │
└──────────────────────────────────┴───────────────────────────────┘
```

### 2.2 Backbone Ablation

```python
# What we have trained:
backbones = {
    "ResNet-50": "outputs/training_resnet50/best_model.pth",
    "EfficientNetV2-M": "outputs/training_efficientnet/best_model.pth",
    "ConvNeXt-Base": "outputs/training_convnext/best_model.pth",  # In training
}

# Plus HuggingFace pretrained:
hf_models = [
    "dima806/deepfake_vs_real_image_detection",
    "umm-maybe/AI-image-detector",
    "Nahrawy/AIorNot",
    "NYUAD-ComNets/NYUAD_AI-generated_images_detector",
]
```

---

## PART III: CALIBRATION & UNCERTAINTY (MANDATORY)

### 3.1 Calibration Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                    CALIBRATION DELIVERABLES                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. Reliability Diagram (Figure 1)                                │
│    - Calibration curve for each method                           │
│    - Perfect calibration line (diagonal)                         │
│    - Histogram of predictions                                    │
│                                                                  │
│ 2. ECE Table (Table 5)                                          │
│    - ECE before calibration                                      │
│    - ECE after temperature scaling                               │
│    - ECE after Platt scaling                                     │
│    - ECE after isotonic regression                               │
│    - Improvement percentage                                      │
│                                                                  │
│ 3. Brier Score Comparison                                        │
│    - Measures calibration + discrimination combined              │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Uncertainty & Abstain Mechanism (MANDATORY)

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNCERTAINTY ANALYSIS                          │
├─────────────────────────────────────────────────────────────────┤
│ Methods Implemented (src/imagetrust/baselines/uncertainty.py):   │
│ ├── Entropy-based uncertainty                                    │
│ ├── Margin-based uncertainty                                     │
│ ├── Confidence-based uncertainty                                 │
│ └── Ensemble variance                                            │
│                                                                  │
│ Deliverables:                                                    │
│ 1. Coverage vs Accuracy Curve (Figure 5)                         │
│    - Trade-off between coverage and accuracy                     │
│    - AURC (Area Under Risk-Coverage Curve)                       │
│                                                                  │
│ 2. Abstain Analysis Table                                        │
│    - Threshold → Coverage → Accuracy on covered samples          │
│    - Error rejection rate                                        │
│                                                                  │
│ 3. UNCERTAIN Verdict Integration                                 │
│    - When the model is unsure → returns "UNCERTAIN"              │
│    - Clearly define the abstain conditions                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## PART IV: CROSS-GENERATOR & DEGRADATION (MANDATORY)

### 4.1 Cross-Generator Evaluation

Test generalization on generators unseen during training:

```
┌─────────────────────────────────────────────────────────────────┐
│                 CROSS-GENERATOR MATRIX                           │
├─────────────────────────────────────────────────────────────────┤
│                    │ Test Generator                              │
│ Train Generator    │ MJ │ DALL-E │ SD-XL │ Firefly │ Real      │
├────────────────────┼────┼────────┼───────┼─────────┼───────────┤
│ Mixed (All)        │ ?? │   ??   │  ??   │   ??    │   ??      │
│ Leave-One-Out      │ ?? │   ??   │  ??   │   ??    │   ??      │
└─────────────────────────────────────────────────────────────────┘

Generators to Test:
├── Midjourney v5/v6
├── DALL-E 2/3
├── Stable Diffusion 1.5/2.1/XL/3.0
├── Adobe Firefly
├── Ideogram
├── Flux
└── Real photos (diverse cameras)
```

### 4.2 Degradation Robustness

Test robustness to post-processing:

```
┌─────────────────────────────────────────────────────────────────┐
│                 DEGRADATION SUITE                                │
├─────────────────────────────────────────────────────────────────┤
│ Degradation Type   │ Parameters                                  │
├────────────────────┼────────────────────────────────────────────┤
│ JPEG Compression   │ Q = 95, 85, 70, 50, 30                      │
│ Resize             │ 75%, 50%, 25% + back to original            │
│ Gaussian Blur      │ σ = 0.5, 1.0, 2.0, 3.0                      │
│ Gaussian Noise     │ σ = 0.01, 0.03, 0.05                        │
│ Brightness/Contrast│ ±20%, ±40%                                  │
│ Crop & Pad         │ 90%, 80%, 70% center crop                   │
│ Screenshot Sim     │ Render in browser + capture                 │
│ WhatsApp Sim       │ JPEG Q=70 + resize to 1600px               │
│ Instagram Sim      │ JPEG Q=80 + slight sharpening              │
└─────────────────────────────────────────────────────────────────┘
```

---

## PART V: NOVEL CONTRIBUTIONS (Screenshot + Social Media Detection)

### 5.1 Screenshot/Recapture Detection (PHASE 8)

**WHAT IS NEW:** An ML detector to identify whether an image is a screenshot/recapture.

```
┌─────────────────────────────────────────────────────────────────┐
│              SCREENSHOT DETECTION MODULE                         │
├─────────────────────────────────────────────────────────────────┤
│ Problem: Screenshot/recaptured images have:                      │
│ ├── Moiré patterns                                               │
│ ├── Color banding                                                │
│ ├── Resolution artifacts                                         │
│ └── UI elements/borders (sometimes)                              │
│                                                                  │
│ Proposed Solution:                                               │
│ 1. Automatic Dataset Generation                                  │
│    ├── Take original images from the dataset                     │
│    ├── Display on screen + screenshot (Selenium/PIL)             │
│    ├── Or: simulate screenshot artifacts programmatically        │
│    └── Label: original=0, screenshot=1                           │
│                                                                  │
│ 2. Train Classifier                                              │
│    ├── Binary classifier (original vs screenshot)                │
│    ├── Uses features: frequency analysis, noise patterns         │
│    └── Lightweight CNN or even SVM on features                   │
│                                                                  │
│ 3. Pipeline Integration                                          │
│    └── Output: "screenshot_probability": 0.0-1.0                 │
│                                                                  │
│ File: src/imagetrust/detection/screenshot_detector.py            │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Social Media Recompression Detection (PHASE 9)

**WHAT IS NEW:** A detector to identify whether an image has been through social media.

```
┌─────────────────────────────────────────────────────────────────┐
│           SOCIAL MEDIA RECOMPRESSION DETECTOR                    │
├─────────────────────────────────────────────────────────────────┤
│ Target Platforms:                                                │
│ ├── WhatsApp (JPEG Q~70, resize to 1600px, strips EXIF)         │
│ ├── Instagram (JPEG Q~80, specific sharpening)                   │
│ ├── Facebook (JPEG Q~85, resize)                                 │
│ ├── Twitter/X (JPEG compression varies)                          │
│ └── Telegram (minimal compression)                               │
│                                                                  │
│ Approach:                                                        │
│ 1. Multi-Class Classification                                    │
│    ├── Class 0: Original (no recompression)                      │
│    ├── Class 1: WhatsApp-like                                    │
│    ├── Class 2: Instagram-like                                   │
│    ├── Class 3: Facebook-like                                    │
│    └── Class 4: Generic social media                             │
│                                                                  │
│ 2. Dataset Generation                                            │
│    ├── Simulate each platform programmatically                   │
│    ├── Document the exact parameters used                        │
│    └── Balance: ~2000 images per class                           │
│                                                                  │
│ 3. Conservative Attribution                                      │
│    ├── Do NOT claim specific platform without high confidence    │
│    ├── Output: "likely_recompressed": true/false                 │
│    ├── Output: "recompression_type": "whatsapp-like" / "unknown" │
│    └── Output: "recompression_confidence": 0.0-1.0               │
│                                                                  │
│ File: src/imagetrust/detection/recompression_detector.py         │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 What Must Be Mentioned in the Publication

```
┌─────────────────────────────────────────────────────────────────┐
│           CONTRIBUTIONS - WHAT IS NEW vs WHAT IS STANDARD        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ ✅ STANDARD (Does not require novelty claim):                    │
│ ├── Ensemble of pretrained detectors                             │
│ ├── Temperature/Platt/Isotonic calibration                       │
│ ├── Cross-generator evaluation protocol                          │
│ ├── Degradation robustness testing                               │
│ ├── Ablation study methodology                                   │
│ └── Uncertainty estimation (entropy, margin)                     │
│                                                                  │
│ ⭐ OWN CONTRIBUTIONS (Must be highlighted):                     │
│ ├── 1. Custom-trained models on specific dataset                 │
│ │      → "We fine-tune ResNet-50, EfficientNetV2-M, and          │
│ │         ConvNeXt-Base on our curated dataset"                  │
│ │                                                                │
│ ├── 2. Multi-signal fusion architecture                          │
│ │      → "We combine ML predictions with frequency and           │
│ │         noise analysis for robust detection"                   │
│ │                                                                │
│ ├── 3. Screenshot/Recapture Detection Module (NEW!)              │
│ │      → "We introduce a novel screenshot detection module       │
│ │         that identifies recaptured images with X% accuracy"    │
│ │                                                                │
│ ├── 4. Social Media Recompression Detection (NEW!)               │
│ │      → "We propose a social media forensics module that        │
│ │         detects platform-specific compression artifacts"       │
│ │                                                                │
│ └── 5. End-to-end Desktop Application                           │
│        → "We provide a complete forensic tool as standalone      │
│           Windows application for practical deployment"          │
│                                                                  │
│ RECOMMENDED PHRASING IN PAPER:                                   │
│ "Our main contributions are:                                     │
│  (1) A comprehensive AI image detection system combining         │
│      multiple pretrained and custom-trained models;              │
│  (2) Novel modules for screenshot and social media              │
│      recompression detection;                                    │
│  (3) Extensive evaluation across generators and degradations;    │
│  (4) A practical desktop application for forensic analysis."    │
└─────────────────────────────────────────────────────────────────┘
```

---

## PART VI: TABLES AND FIGURES FOR THE PAPER

### 6.1 Required Tables

| Table # | Content | Status |
|---------|---------|--------|
| Table 1 | Main Comparison: Baselines vs ImageTrust (Acc, Bal.Acc, Prec, Rec, F1, AUC, ECE) | ⬜ |
| Table 2 | Cross-Generator Performance Matrix | ⬜ |
| Table 3 | Degradation Robustness Results | ⬜ |
| Table 4 | Ablation Study Results | ⬜ |
| Table 5 | Calibration Comparison (ECE before/after) | ⬜ |
| Table 6 | Efficiency Metrics (ms/img, throughput, VRAM) | ⬜ |
| Table 7 | Screenshot Detection Results (NEW) | ⬜ |
| Table 8 | Social Media Recompression Results (NEW) | ⬜ |
| Table 9 | Statistical Significance Tests | ⬜ |
| Table 10 | Dataset Statistics | ⬜ |

### 6.2 Required Figures

| Figure # | Content | Status |
|----------|---------|--------|
| Figure 1 | System Architecture Diagram | ⬜ |
| Figure 2 | Reliability Diagram (Calibration Curves) | ⬜ |
| Figure 3 | ROC Curves (per baseline) | ⬜ |
| Figure 4 | Cross-Generator Heatmap | ⬜ |
| Figure 5 | Degradation Performance Curves | ⬜ |
| Figure 6 | Coverage vs Accuracy (Uncertainty) | ⬜ |
| Figure 7 | Grad-CAM Visualizations | ⬜ |
| Figure 8 | UI Screenshots | ⬜ |
| Figure 9 | Confusion Matrices | ⬜ |

---

## PART VII: DESKTOP APPLICATION

### 7.1 UI Requirements

```
┌─────────────────────────────────────────────────────────────────┐
│                    UI SPECIFICATIONS                             │
├─────────────────────────────────────────────────────────────────┤
│ Framework: PySide6 (Qt) - Professional, native look             │
│                                                                  │
│ Main Window Layout:                                              │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ [Logo] ImageTrust - AI Image Forensics          [─][□][×]   │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ ┌─────────────────────┐ ┌─────────────────────────────────┐ │ │
│ │ │                     │ │ ANALYSIS RESULTS                │ │ │
│ │ │   [Drag & Drop]     │ │                                 │ │ │
│ │ │                     │ │ Verdict: AI-GENERATED           │ │ │
│ │ │   Image Preview     │ │ Confidence: 94.2%               │ │ │
│ │ │                     │ │                                 │ │ │
│ │ │   512×512 preview   │ │ ┌─────────────────────────────┐ │ │ │
│ │ │                     │ │ │ Individual Model Results    │ │ │ │
│ │ │                     │ │ │ ├─ Deepfake-vs-Real: 96.1%  │ │ │ │
│ │ │                     │ │ │ ├─ AI-Detector: 92.3%       │ │ │ │
│ │ │                     │ │ │ ├─ AIorNot: 94.8%           │ │ │ │
│ │ │                     │ │ │ └─ NYUAD: 93.5%             │ │ │ │
│ │ │                     │ │ └─────────────────────────────┘ │ │ │
│ │ └─────────────────────┘ │                                 │ │ │
│ │                         │ Signal Analysis:                │ │ │
│ │ [Analyze] [Export PDF]  │ ├─ Frequency: 0.72 (AI-like)    │ │ │
│ │                         │ └─ Noise: 0.65 (AI-like)        │ │ │
│ │                         │                                 │ │ │
│ │                         │ Forensics:                      │ │ │
│ │                         │ ├─ Screenshot: NO (12%)         │ │ │
│ │                         │ └─ Social Media: YES (78%)      │ │ │
│ │                         └─────────────────────────────────┘ │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ Status: Analysis complete │ Time: 1.2s │ GPU: RTX 5080     │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

Additional Tabs/Features:
├── Metadata Tab: EXIF, XMP, C2PA provenance
├── Explainability Tab: Grad-CAM heatmaps
├── Batch Processing: Analyze folder
├── Report Export: PDF with all findings
└── Settings: GPU selection, threshold tuning
```

### 7.2 Build & Distribution

```
Build Process:
├── PyInstaller for .exe
├── Include model weights (or download on first-run)
├── Code signing (optional, for Windows SmartScreen)
└── Installer with NSIS (optional)

Testing Checklist:
├── [ ] Works on Windows 10/11 clean install
├── [ ] Works without NVIDIA GPU (CPU fallback)
├── [ ] Works offline
├── [ ] PDF export works
├── [ ] Handles all image formats (JPEG, PNG, WebP, HEIC)
└── [ ] Memory usage reasonable (<4GB RAM)
```

---

## PART VIII: DETAILED TIMELINE

### After ConvNeXt finishes (~11h remaining at 81h total):

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION TIMELINE                            │
├──────┬──────────────────────────────────────────┬───────────────┤
│ Day  │ Task                                     │ Hours         │
├──────┼──────────────────────────────────────────┼───────────────┤
│ D+0  │ ConvNeXt finishes training               │ -             │
│      │ Verify ConvNeXt results                  │ 0.5h          │
│      │ Fix recall issue (threshold tuning)      │ 2h            │
├──────┼──────────────────────────────────────────┼───────────────┤
│ D+1  │ Classical Baseline (XGBoost)             │ 3h            │
│      │ Run full baseline comparison             │ 2h            │
│      │ Generate Table 1 (Main Results)          │ 1h            │
├──────┼──────────────────────────────────────────┼───────────────┤
│ D+2  │ Ablation Study Complete                  │ 3h            │
│      │ Generate Table 4 (Ablation)              │ 1h            │
│      │ Calibration Analysis                     │ 2h            │
│      │ Generate Table 5 + Figure 2              │ 1h            │
├──────┼──────────────────────────────────────────┼───────────────┤
│ D+3  │ Cross-Generator Evaluation               │ 3h            │
│      │ Generate Table 2 + Figure 4              │ 1h            │
│      │ Degradation Testing                      │ 3h            │
│      │ Generate Table 3 + Figure 5              │ 1h            │
├──────┼──────────────────────────────────────────┼───────────────┤
│ D+4  │ Screenshot Detection - Dataset Gen       │ 4h            │
│      │ Screenshot Detection - Training          │ 4h            │
├──────┼──────────────────────────────────────────┼───────────────┤
│ D+5  │ Screenshot Detection - Eval + Table 7    │ 2h            │
│      │ Social Media Detection - Dataset Gen     │ 4h            │
│      │ Social Media Detection - Training        │ 4h            │
├──────┼──────────────────────────────────────────┼───────────────┤
│ D+6  │ Social Media Detection - Eval + Table 8  │ 2h            │
│      │ Uncertainty Analysis + Figure 6          │ 2h            │
│      │ Efficiency Profiling + Table 6           │ 2h            │
├──────┼──────────────────────────────────────────┼───────────────┤
│ D+7  │ UI Implementation - Core                 │ 8h            │
├──────┼──────────────────────────────────────────┼───────────────┤
│ D+8  │ UI Implementation - Polish               │ 8h            │
│      │ UI Screenshots for the paper             │ 1h            │
├──────┼──────────────────────────────────────────┼───────────────┤
│ D+9  │ Build .exe                               │ 4h            │
│      │ Test on clean Windows VM                 │ 2h            │
│      │ Fix issues                               │ 2h            │
├──────┼──────────────────────────────────────────┼───────────────┤
│ D+10 │ Final Documentation                      │ 4h            │
│      │ README, User Guide                       │ 2h            │
│      │ Code cleanup                             │ 2h            │
├──────┼──────────────────────────────────────────┼───────────────┤
│      │ TOTAL                                    │ ~80h          │
└──────┴──────────────────────────────────────────┴───────────────┘
```

---

## PART IX: COMMANDS REFERENCE

### After ConvNeXt, run in order:

```bash
# 1. Verify ConvNeXt results
python scripts/evaluate_model.py --model outputs/training_convnext/best_model.pth

# 2. Fix recall (adjust threshold)
python scripts/optimize_threshold.py --models outputs/training_*/best_model.pth

# 3. Run baseline comparison
python scripts/run_baselines.py --dataset data/test --output outputs/baselines

# 4. Run ablation study
python scripts/run_ablation.py --dataset data/test --output outputs/ablation

# 5. Cross-generator evaluation
python scripts/run_evaluation.py --dataset data/eval --cross-generator

# 6. Degradation evaluation
python scripts/run_evaluation.py --dataset data/eval --degradation

# 7. Generate tables
python scripts/generate_tables.py --results outputs --output reports/tables

# 8. Generate figures
python scripts/generate_figures.py --results outputs --output reports/figures

# 9. Build .exe
powershell -File scripts/build_exe.ps1
```

---

## PART X: FINAL CHECKLIST

### Before Submission:

```
Academic Requirements:
├── [ ] Table 1: Main comparison (≥3 baselines + ImageTrust)
├── [ ] Table 2: Cross-generator matrix
├── [ ] Table 3: Degradation robustness
├── [ ] Table 4: Ablation study
├── [ ] Table 5: Calibration (ECE before/after)
├── [ ] Table 6: Efficiency metrics
├── [ ] Figure 1: Architecture diagram
├── [ ] Figure 2: Reliability diagram
├── [ ] Figure 3: ROC curves
├── [ ] Figure 4: Cross-generator heatmap
├── [ ] Figure 5: Degradation curves
├── [ ] Reproducibility: seed=42, hardware specs documented

Novel Contributions:
├── [ ] Screenshot detection module + Table 7
├── [ ] Social media recompression detector + Table 8
├── [ ] Working desktop application

Deliverables:
├── [ ] Paper draft complete
├── [ ] Code repository clean
├── [ ] .exe tested on 3+ machines
├── [ ] User documentation
└── [ ] Demo video (optional)
```

---

## Final Note

This plan puts you at **international conference B level**.

**What sets you apart:**
1. Extremely rigorous evaluation (cross-generator, degradation, ablation)
2. Novel modules (screenshot + social media) - original contributions
3. Functional practical application
4. Complete reproducibility

**Risks:**
- Low recall on models (~47%) - **MUST BE FIXED**
- Limited time for UI polish
- .exe testing may have surprises

Let me know when ConvNeXt finishes and we start the execution!
