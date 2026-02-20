# ImageTrust - Master Plan Academic
## Disertație de Master - Nivel Conferință Internațională B (IEEE WIFS / ACM IH&MMSec)

**Data:** Februarie 2026
**Status:** ConvNeXt în antrenare (Epoch 14/25)

---

## Executive Summary

| Fază | Descriere | Timp Estimat | Status |
|------|-----------|--------------|--------|
| 1 | Setup + Antrenare Modele Custom | ~150h | ✅ 85% Done |
| 2 | Evaluare Academică Completă | 4-6h | ⬜ Next |
| 3 | Ablation Study Complet | 2-3h | ⬜ |
| 4 | Calibrare + Uncertainty | 1-2h | ⬜ |
| 5 | Screenshot/Recapture Detection | 8-12h | ⬜ NOU |
| 6 | Social Media Recompression | 8-12h | ⬜ NOU |
| 7 | Cross-Generator Evaluation | 2-3h | ⬜ |
| 8 | Degradation Robustness | 2-3h | ⬜ |
| 9 | Tabele LaTeX + Figuri PDF | 2-4h | ⬜ |
| 10 | UI Professional (PySide6) | 16-24h | ⬜ |
| 11 | Build .exe + Testing | 4-8h | ⬜ |
| 12 | Documentație + README Final | 2-4h | ⬜ |

**Total rămas:** ~50-80h de muncă după ConvNeXt

---

## PARTEA I: FUNDAMENT ACADEMIC (Obligatoriu pentru Publicare)

### 1.1 Modele Antrenate Custom (✅ În Progres)

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

**Observație Critică:** Recall scăzut (~47%) - necesită threshold tuning sau focal loss.

### 1.2 Baseline Comparison (OBLIGATORIU)

Conform CLAUDE.md, trebuie comparație cu **minimum 3 baseline-uri**:

| Baseline | Metodă | Implementare | Status |
|----------|--------|--------------|--------|
| **Baseline 1** | Classical ML | XGBoost pe features forensice (JPEG artifacts, noise, frequency) | ⬜ TODO |
| **Baseline 2** | CNN Single | ResNet-50 fine-tuned (cel antrenat de tine) | ✅ Done |
| **Baseline 3** | Modern/ViT | HuggingFace pretrained detectors (ensemble din 4 modele) | ✅ Done |
| **ImageTrust** | Ensemble + Signal Analysis | Custom trained + HF models + forensics | ✅ Done |

### 1.3 Metrics Required (Paper-Ready)

Pentru fiecare metodă, trebuie raportate:

```
Primary Metrics (Table 1):
├── Accuracy
├── Balanced Accuracy  ← ADĂUGAT în metrics.py
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

## PARTEA II: ABLATION STUDY (OBLIGATORIU)

### 2.1 Component Ablation

Testează impactul fiecărei componente:

```
┌──────────────────────────────────────────────────────────────────┐
│                      ABLATION EXPERIMENTS                         │
├──────────────────────────────────┬───────────────────────────────┤
│ Experiment                       │ Ce Măsoară                    │
├──────────────────────────────────┼───────────────────────────────┤
│ Full System (Baseline)           │ Performanța completă          │
│ - Without Model 1 (Deepfake)     │ Contribuția modelului 1       │
│ - Without Model 2 (AI-Detector)  │ Contribuția modelului 2       │
│ - Without Model 3 (AIorNot)      │ Contribuția modelului 3       │
│ - Without Model 4 (NYUAD)        │ Contribuția modelului 4       │
│ - Without Signal Analysis        │ Valoarea analizei de semnal   │
│ - Without Calibration            │ Impactul calibrării           │
│ - Single Best Model Only         │ Beneficiul ensemble           │
├──────────────────────────────────┼───────────────────────────────┤
│ Ensemble Strategy Comparison     │                               │
├──────────────────────────────────┼───────────────────────────────┤
│ Average Voting                   │ Strategie simplă              │
│ Weighted Voting                  │ Cu ponderi pe model           │
│ Majority Voting                  │ Vot majoritar                 │
│ Max Probability                  │ Cel mai confident             │
│ Median                           │ Robust la outlieri            │
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
# Ce avem antrenat:
backbones = {
    "ResNet-50": "outputs/training_resnet50/best_model.pth",
    "EfficientNetV2-M": "outputs/training_efficientnet/best_model.pth",
    "ConvNeXt-Base": "outputs/training_convnext/best_model.pth",  # În antrenare
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

## PARTEA III: CALIBRATION & UNCERTAINTY (OBLIGATORIU)

### 3.1 Calibration Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                    CALIBRATION DELIVERABLES                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. Reliability Diagram (Figure 1)                                │
│    - Calibration curve pentru fiecare metodă                     │
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
│    - Trade-off între coverage și accuracy                        │
│    - AURC (Area Under Risk-Coverage Curve)                       │
│                                                                  │
│ 2. Abstain Analysis Table                                        │
│    - Threshold → Coverage → Accuracy pe covered samples          │
│    - Error rejection rate                                        │
│                                                                  │
│ 3. UNCERTAIN Verdict Integration                                 │
│    - Când modelul nu e sigur → returnează "UNCERTAIN"            │
│    - Definește clar condițiile de abstain                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## PARTEA IV: CROSS-GENERATOR & DEGRADATION (OBLIGATORIU)

### 4.1 Cross-Generator Evaluation

Testează generalizarea pe generatori ne-văzuți în training:

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

Testează robustețea la post-procesare:

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

## PARTEA V: NOI CONTRIBUȚII (Screenshot + Social Media Detection)

### 5.1 Screenshot/Recapture Detection (FAZA 8)

**CE E NOU:** Detector ML pentru a identifica dacă o imagine e screenshot/recapture.

```
┌─────────────────────────────────────────────────────────────────┐
│              SCREENSHOT DETECTION MODULE                         │
├─────────────────────────────────────────────────────────────────┤
│ Problemă: Imaginile screenshot/recaptured au:                    │
│ ├── Moiré patterns                                               │
│ ├── Color banding                                                │
│ ├── Resolution artifacts                                         │
│ └── UI elements/borders (sometimes)                              │
│                                                                  │
│ Soluție Propusă:                                                 │
│ 1. Generare Dataset Automat                                      │
│    ├── Ia imagini originale din dataset                          │
│    ├── Afișează pe ecran + screenshot (Selenium/PIL)             │
│    ├── Sau: simulează screenshot artifacts programatic           │
│    └── Etichetează: original=0, screenshot=1                     │
│                                                                  │
│ 2. Train Classifier                                              │
│    ├── Binary classifier (original vs screenshot)                │
│    ├── Folosește features: frequency analysis, noise patterns    │
│    └── Lightweight CNN sau even SVM pe features                  │
│                                                                  │
│ 3. Integrare în Pipeline                                         │
│    └── Output: "screenshot_probability": 0.0-1.0                 │
│                                                                  │
│ Fișier: src/imagetrust/detection/screenshot_detector.py          │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Social Media Recompression Detection (FAZA 9)

**CE E NOU:** Detector pentru a identifica dacă imaginea a trecut prin social media.

```
┌─────────────────────────────────────────────────────────────────┐
│           SOCIAL MEDIA RECOMPRESSION DETECTOR                    │
├─────────────────────────────────────────────────────────────────┤
│ Platforme Țintă:                                                 │
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
│    ├── Simulează fiecare platformă programatic                   │
│    ├── Documentează exact parametrii folosiți                    │
│    └── Balance: ~2000 imagini per clasă                          │
│                                                                  │
│ 3. Conservative Attribution                                      │
│    ├── NU claim specific platform fără high confidence           │
│    ├── Output: "likely_recompressed": true/false                 │
│    ├── Output: "recompression_type": "whatsapp-like" / "unknown" │
│    └── Output: "recompression_confidence": 0.0-1.0               │
│                                                                  │
│ Fișier: src/imagetrust/detection/recompression_detector.py       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Ce Trebuie Menționat în Publicație

```
┌─────────────────────────────────────────────────────────────────┐
│           CONTRIBUȚII - CE E NOU vs CE E STANDARD                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ ✅ STANDARD (Nu necesită claim de noutate):                      │
│ ├── Ensemble of pretrained detectors                             │
│ ├── Temperature/Platt/Isotonic calibration                       │
│ ├── Cross-generator evaluation protocol                          │
│ ├── Degradation robustness testing                               │
│ ├── Ablation study methodology                                   │
│ └── Uncertainty estimation (entropy, margin)                     │
│                                                                  │
│ ⭐ CONTRIBUȚII PROPRII (Trebuie evidențiate):                    │
│ ├── 1. Custom-trained models pe dataset specific                 │
│ │      → "We fine-tune ResNet-50, EfficientNetV2-M, and          │
│ │         ConvNeXt-Base on our curated dataset"                  │
│ │                                                                │
│ ├── 2. Multi-signal fusion architecture                          │
│ │      → "We combine ML predictions with frequency and           │
│ │         noise analysis for robust detection"                   │
│ │                                                                │
│ ├── 3. Screenshot/Recapture Detection Module (NOU!)             │
│ │      → "We introduce a novel screenshot detection module       │
│ │         that identifies recaptured images with X% accuracy"    │
│ │                                                                │
│ ├── 4. Social Media Recompression Detection (NOU!)              │
│ │      → "We propose a social media forensics module that        │
│ │         detects platform-specific compression artifacts"       │
│ │                                                                │
│ └── 5. End-to-end Desktop Application                           │
│        → "We provide a complete forensic tool as standalone      │
│           Windows application for practical deployment"          │
│                                                                  │
│ 📝 FORMULARE RECOMANDATĂ ÎN PAPER:                              │
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

## PARTEA VI: TABELE ȘI FIGURI PENTRU PAPER

### 6.1 Tabele Obligatorii

| Table # | Conținut | Status |
|---------|----------|--------|
| Table 1 | Main Comparison: Baselines vs ImageTrust (Acc, Bal.Acc, Prec, Rec, F1, AUC, ECE) | ⬜ |
| Table 2 | Cross-Generator Performance Matrix | ⬜ |
| Table 3 | Degradation Robustness Results | ⬜ |
| Table 4 | Ablation Study Results | ⬜ |
| Table 5 | Calibration Comparison (ECE before/after) | ⬜ |
| Table 6 | Efficiency Metrics (ms/img, throughput, VRAM) | ⬜ |
| Table 7 | Screenshot Detection Results (NOU) | ⬜ |
| Table 8 | Social Media Recompression Results (NOU) | ⬜ |
| Table 9 | Statistical Significance Tests | ⬜ |
| Table 10 | Dataset Statistics | ⬜ |

### 6.2 Figuri Obligatorii

| Figure # | Conținut | Status |
|----------|----------|--------|
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

## PARTEA VII: DESKTOP APPLICATION

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
├── PyInstaller pentru .exe
├── Include model weights (sau download la first-run)
├── Code signing (opțional, pentru Windows SmartScreen)
└── Installer cu NSIS (opțional)

Testing Checklist:
├── [ ] Works on Windows 10/11 clean install
├── [ ] Works without NVIDIA GPU (CPU fallback)
├── [ ] Works offline
├── [ ] PDF export works
├── [ ] Handles all image formats (JPEG, PNG, WebP, HEIC)
└── [ ] Memory usage reasonable (<4GB RAM)
```

---

## PARTEA VIII: TIMELINE DETALIATĂ

### După terminarea ConvNeXt (~11h rămase la 81h total):

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
│      │ UI Screenshots pentru paper              │ 1h            │
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

## PARTEA IX: COMMANDS REFERENCE

### După ConvNeXt, rulează în ordine:

```bash
# 1. Verifică rezultatele ConvNeXt
python scripts/evaluate_model.py --model outputs/training_convnext/best_model.pth

# 2. Fix recall (ajustează threshold)
python scripts/optimize_threshold.py --models outputs/training_*/best_model.pth

# 3. Rulează baseline comparison
python scripts/run_baselines.py --dataset data/test --output outputs/baselines

# 4. Rulează ablation study
python scripts/run_ablation.py --dataset data/test --output outputs/ablation

# 5. Evaluare cross-generator
python scripts/run_evaluation.py --dataset data/eval --cross-generator

# 6. Evaluare degradation
python scripts/run_evaluation.py --dataset data/eval --degradation

# 7. Generează tabele
python scripts/generate_tables.py --results outputs --output reports/tables

# 8. Generează figuri
python scripts/generate_figures.py --results outputs --output reports/figures

# 9. Build .exe
powershell -File scripts/build_exe.ps1
```

---

## PARTEA X: CHECKLIST FINAL

### Înainte de Submission:

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

## Notă Finală

Acest plan te pune la nivel de **conferință internațională B**.

**Ce te diferențiază:**
1. Evaluare extrem de riguroasă (cross-generator, degradation, ablation)
2. Module noi (screenshot + social media) - contribuții originale
3. Aplicație practică funcțională
4. Reproducibilitate completă

**Riscuri:**
- Recall scăzut la modele (~47%) - **TREBUIE FIXAT**
- Timp limitat pentru UI polish
- Testare .exe poate avea surprize

Spune-mi când termină ConvNeXt și începem execuția!
