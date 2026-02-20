# Threshold Calibration - Methodological Guide

**Professor's requirement:** Each threshold must be justified theoretically and experimentally, not "set arbitrarily".

---

## 1. The Problem

Current thresholds without justification:
```python
# Examples of UNJUSTIFIED values in code:
ELA_THRESHOLD = 0.72        # Where from?
NOISE_THRESHOLD = 0.5       # Where from?
AI_CONFIDENCE_HIGH = 0.85   # Where from?
```

**What the professor expects:**
- Theoretical basis (papers)
- Experimental validation (dataset)
- Comparison section for each method

---

## 2. Calibration Framework for Each Method

### Structure for each forensic technique:

```
1. THEORY
   - What does the method detect?
   - Reference papers (minimum 3)
   - What do the papers say about thresholds?

2. EXPERIMENT
   - Dataset used (with ground truth)
   - Metrics calculated (ROC curve, F1)
   - Optimal threshold found

3. VALIDATION
   - Test on a separate dataset
   - Comparison with thresholds from the literature
   - Error analysis (false positives/negatives)
```

---

## 3. ELA (Error Level Analysis)

### 3.1 Theory

**What it detects:**
ELA re-compresses the image at a known JPEG level (e.g., Q95) and compares it with the original. Recently edited regions have different errors compared to the original ones.

**Reference papers:**

1. **Krawetz, N. (2007)** - "A Picture's Worth... Digital Image Analysis and Forensics"
   - Hack in the Box Conference
   - Introduces ELA for the first time
   - Does NOT provide a numeric threshold, says "look for inconsistencies"
   - Link: https://www.hackerfactor.com/papers/bh-usa-07-krawetz-wp.pdf

2. **Gunawan, T.S. et al. (2017)** - "Development of Photo Forensic using ELA"
   - International Journal of Electrical and Computer Engineering
   - Tested on a dataset of 200 images
   - Threshold varied between 15-30 (in pixel space, not 0-1)
   - DOI: 10.11591/ijece.v7i5.pp2690-2695

3. **Sudiatmika, I.B. et al. (2019)** - "Image Forgery Detection Using ELA and CNN"
   - International Conference on Data Science and Information Technology
   - Combines ELA with CNN
   - ELA as preprocessing, no direct threshold

**Theoretical conclusion:**
The papers do NOT provide a universal threshold. It must be calibrated on a dataset.

### 3.2 Experiment - How to Calibrate

```python
# Pseudocode for ELA threshold calibration

import numpy as np
from sklearn.metrics import roc_curve, f1_score

def calibrate_ela_threshold(dataset_path, ground_truth):
    """
    dataset_path: folder with images
    ground_truth: dict {image_name: 0=authentic, 1=manipulated}
    """

    scores = []
    labels = []

    for image_path in dataset_path:
        # Calculate ELA score (normalized 0-1)
        ela_score = compute_ela_score(image_path)
        scores.append(ela_score)
        labels.append(ground_truth[image_path.name])

    scores = np.array(scores)
    labels = np.array(labels)

    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Method 1: Youden's J statistic (maximizes TPR - FPR)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold_youden = thresholds[optimal_idx]

    # Method 2: Maximize F1
    f1_scores = []
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds = (scores >= thresh).astype(int)
        f1 = f1_score(labels, preds)
        f1_scores.append((thresh, f1))

    optimal_threshold_f1 = max(f1_scores, key=lambda x: x[1])[0]

    return {
        'threshold_youden': optimal_threshold_youden,
        'threshold_f1': optimal_threshold_f1,
        'auc': auc(fpr, tpr),
        'n_samples': len(scores)
    }
```

### 3.3 Datasets for ELA Validation

| Dataset | Images | Type | Link |
|---------|--------|------|------|
| CASIA v2.0 | 12,614 | Splicing, Copy-Move | http://forensics.idealtest.org |
| Columbia | 1,845 | Splicing | https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/ |
| COVERAGE | 100 pairs | Copy-Move | https://github.com/wenbihan/coverage |
| IMD2020 | 35,000 | Mixed manipulation | https://staff.fnwi.uva.nl/z.geradts/forensicimagetools/ |

### 3.4 Expected Results (Example)

```
Dataset: CASIA v2.0
Images tested: 5,123
Authentic: 2,500 | Manipulated: 2,623

Calibration results:
- Optimal threshold (Youden): 0.68
- Optimal threshold (F1): 0.71
- AUC: 0.823
- F1 at threshold 0.71: 0.79

Conclusion: ELA Threshold = 0.70 (rounded)
Justification: Optimized for F1 on CASIA v2.0
```

---

## 4. Noise Inconsistency Analysis

### 4.1 Theory

**What it detects:**
Digital cameras introduce characteristic noise. Regions from different sources have different noise patterns.

**Reference papers:**

1. **Mahdian, B. & Saic, S. (2009)** - "Using noise inconsistencies for blind image forensics"
   - Image and Vision Computing, Vol 27
   - Proposes noise variance analysis on blocks
   - DOI: 10.1016/j.imavis.2008.06.010

2. **Pan, X. et al. (2011)** - "Region Duplication Detection Using Image Feature Matching"
   - IEEE Transactions on Information Forensics and Security
   - Noise + features for copy-move

3. **Cozzolino, D. et al. (2015)** - "Efficient dense-field copy-move forgery detection"
   - IEEE Transactions on Information Forensics and Security
   - State-of-the-art noise analysis

### 4.2 Calibration

Similar to ELA:
1. Dataset with ground truth (manipulated regions marked)
2. Calculate noise variance per block
3. Find threshold for "significant inconsistency"
4. Validate on a separate dataset

---

## 5. JPEG Artifacts (Double JPEG Detection)

### 5.1 Theory

**What it detects:**
The first JPEG compression leaves "ghosts" in the DCT coefficient histogram. Re-compression creates a detectable periodic pattern.

**Reference papers:**

1. **Farid, H. (2009)** - "Exposing Digital Forgeries from JPEG Ghosts"
   - IEEE Transactions on Information Forensics and Security
   - Introduces "JPEG ghosts"
   - DOI: 10.1109/TIFS.2008.2012215

2. **Bianchi, T. & Piva, A. (2012)** - "Image Forgery Localization via Block-Grained Analysis"
   - IEEE Transactions on Information Forensics and Security
   - Per-block DCT analysis

3. **Li, W. et al. (2015)** - "Passive Detection of Doctored JPEG Image via Block Artifact Grid Extraction"
   - Signal Processing
   - Grid artifacts detection

### 5.2 Calibration

```python
def calibrate_jpeg_threshold(dataset):
    """
    For double-JPEG detection:
    - 0 = single compression (or converted PNG)
    - 1 = multiple compressions detected
    """
    # Calculation based on periodicity in the DCT histogram
    # Threshold = how "strong" the periodicity is
```

---

## 6. AI Detection (HuggingFace Models)

### 6.1 Theory

**What it detects:**
CNN/ViT models trained on real/AI pairs learn patterns specific to synthetic images.

**Papers for the models used:**

1. **umm-maybe/AI-image-detector**
   - Based on: Wang et al. "CNN-generated images are surprisingly easy to spot... for now"
   - CVPR 2020
   - https://arxiv.org/abs/1912.11035

2. **Organika/sdxl-detector**
   - Specific to Stable Diffusion XL
   - Fine-tuned on SDXL outputs

3. **Corvi et al. (2023)** - "On The Detection of Synthetic Images Generated by Diffusion Models"
   - ICASSP 2023
   - Analyzes detectability of diffusion models

### 6.2 Calibration

The HF models already return probabilities. However, one must:
1. Check if they are calibrated (Expected Calibration Error)
2. Apply Temperature Scaling if not
3. Set thresholds for HIGH/MEDIUM/LOW confidence

```python
def calibrate_ai_detector_threshold(model, dataset):
    """
    Dataset with AI and real images, labeled
    """
    probs = model.predict_proba(dataset.images)
    labels = dataset.labels  # 0=real, 1=AI

    # Calculate ECE (Expected Calibration Error)
    ece = compute_ece(probs, labels, n_bins=10)

    if ece > 0.05:  # Poorly calibrated
        # Apply temperature scaling
        temperature = find_optimal_temperature(probs, labels)
        calibrated_probs = softmax(logits / temperature)

    # Find thresholds for confidence levels
    # HIGH = minimize false positives (do not wrongly accuse)
    # LOW = minimize false negatives (do not miss AI-generated images)
```

---

## 7. Comparison Chapter Structure (Thesis)

For EACH method, write:

```markdown
## 4.X [Method Name] - Error Level Analysis

### 4.X.1 Theoretical Foundation
- What physical/mathematical principle is at the basis
- When it was proposed, by whom
- Relevant equations

### 4.X.2 Reference Papers
- [Paper 1] - what it says
- [Paper 2] - what it says
- [Paper 3] - what it says
- Thresholds reported in the literature (if any)

### 4.X.3 Implementation in ImageTrust
- Algorithm used
- Configurable parameters
- Differences from standard implementations

### 4.X.4 Experimental Calibration
- Dataset used: [name, size, source]
- Metrics calculated: [AUC, F1, precision, recall]
- Optimal threshold found: X.XX
- Selection method: [Youden's J / Max F1 / other method]

### 4.X.5 Results and Discussion
- Results table
- ROC curve
- Comparison with thresholds from the literature
- Identified limitations

### 4.X.6 Conclusion
- Final adopted threshold: X.XX
- Justification in 2-3 sentences
```

---

## 8. Recommended Datasets for Calibration

### For Traditional Manipulation (ELA, Noise, JPEG):

| Dataset | N | Type | Download |
|---------|---|------|----------|
| **CASIA v1.0** | 1,725 | Splicing | forensics.idealtest.org |
| **CASIA v2.0** | 12,614 | Splicing + CM | forensics.idealtest.org |
| **Columbia** | 1,845 | Splicing | columbia.edu |
| **COVERAGE** | 100 | Copy-Move | github |
| **CoMoFoD** | 10,000 | Copy-Move | comofod.org |
| **GRIP** | 80 | Retouching | github |

### For AI Detection:

| Dataset | N | Generators | Download |
|---------|---|------------|----------|
| **GenImage** | 1.3M | SD, Midjourney, DALL-E, etc. | github.com/GenImage-Dataset |
| **CIFAKE** | 120K | SD 1.4 | kaggle |
| **AI-ArtBench** | 180K | Multiple | huggingface |
| **DiffusionDB** | 14M | SD | poloclub.github.io/diffusiondb |

---

## 9. Final Checklist

For each method (ELA, Noise, JPEG, AI Detection, etc.):

- [ ] Found at least 3 reference papers
- [ ] Chose a dataset with ground truth
- [ ] Ran the calibration and obtained the optimal threshold
- [ ] Calculated AUC and F1 at that threshold
- [ ] Validated on a separate dataset (not the same as calibration)
- [ ] Wrote the thesis section with all justifications
- [ ] Compared my threshold with what the literature reports

---

## 10. Example Final Table for the Thesis

| Method | Threshold | Threshold Source | AUC | F1 | Calibration Dataset |
|--------|-----------|------------------|-----|----|--------------------|
| ELA | 0.70 | Experimentally calibrated | 0.82 | 0.79 | CASIA v2.0 |
| Noise | 0.55 | Experimentally calibrated | 0.75 | 0.71 | CASIA v2.0 |
| JPEG Double | 0.60 | Farid (2009) + calibration | 0.88 | 0.84 | Columbia |
| AI Detection | 0.85 (HIGH) | Calibrated + Temperature | 0.94 | 0.91 | GenImage |

**Note:** Each value has its justification in the corresponding section.

---

*Document created to address the professor's requirements regarding threshold justification.*
