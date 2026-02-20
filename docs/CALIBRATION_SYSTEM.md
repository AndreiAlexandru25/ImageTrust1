# Threshold Calibration System

## Overview

The calibration system implemented in ImageTrust provides rigorous statistical validation for all thresholds used in forensic methods, in accordance with the academic requirements for a Master's thesis.

**Professor's requirement:** "Do not use arbitrary values, demonstrate them experimentally!"

---

## Main Files

| File | Description |
|------|-------------|
| `scripts/calibrate_thresholds.py` | Basic calibration script |
| `scripts/calibrate_thresholds_advanced.py` | **Advanced script** with all features |
| `scripts/generate_calibration_plots.py` | Figure generation for the thesis |
| `docs/THRESHOLD_CALIBRATION.md` | Methodological guide |

---

## Advanced Features

### 1. Threshold Selection Methods

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Youden's J** | Maximizes TPR - FPR | Balanced dataset |
| **F1 Max** | Maximizes F1 score | Moderately imbalanced dataset |
| **F2 Max** | Favors recall | When false negatives are costly |
| **F0.5 Max** | Favors precision | When false positives are costly |
| **MCC Max** | Maximizes Matthews Correlation | Highly imbalanced dataset |
| **G-Mean** | √(TPR × TNR) | Important minority class |
| **EER** | FPR = FNR | Biometric applications |
| **Cost-Sensitive** | Minimizes weighted cost | Different costs for errors |

### 2. Cross-Validation

```bash
python scripts/calibrate_thresholds_advanced.py \
    --dataset data/calibration \
    --k-folds 5 \
    --output outputs/calibration
```

- Stratified K-Fold (preserves class proportions)
- Threshold is found on training, evaluated on validation
- Reports mean ± std for all metrics

### 3. Bootstrap Confidence Intervals

- **BCa (Bias-Corrected and Accelerated)**: More robust than simple percentile
- 95% and 99% intervals
- Default: 1000 resamples

```python
# Example result
threshold_optimal: 0.68
confidence_interval_95: [0.62, 0.74]
confidence_interval_99: [0.59, 0.77]
cv_threshold_std: 0.03
```

### 4. Probability Calibration

#### Temperature Scaling
```
P_calibrated = sigmoid(logits / T)
```
- Optimizes T to minimize Negative Log Likelihood
- Typical: T ∈ [0.5, 2.0]

#### Calibration Metrics
- **ECE (Expected Calibration Error)**: Average calibration error
- **MCE (Maximum Calibration Error)**: Worst-case calibration
- **Brier Score**: Mean squared error for probabilities

### 5. Statistical Tests

#### McNemar Test
- Compares two classifiers on the same dataset
- Tests whether the difference in errors is statistically significant

#### DeLong Test
- Compares the AUCs of two models
- Includes confidence interval for the AUC difference

#### Wilcoxon Signed-Rank
- Compares performance across cross-validation folds
- Non-parametric, does not assume normality

---

## Usage

### Full Calibration

```bash
# Calibration for all methods
python scripts/calibrate_thresholds_advanced.py \
    --dataset data/casia_v2 \
    --dataset-name "CASIA v2.0" \
    --output outputs/calibration \
    --k-folds 5 \
    --n-bootstrap 1000 \
    --seed 42

# Only a single method
python scripts/calibrate_thresholds_advanced.py \
    --dataset data/casia_v2 \
    --method ela \
    --output outputs/calibration_ela
```

### Figure Generation

```bash
python scripts/generate_calibration_plots.py \
    --results outputs/calibration \
    --output outputs/figures \
    --format pdf
```

### Generated Figures

1. **Reliability Diagrams** - Calibration curves
2. **ROC Curves** - Comparison between methods
3. **Precision-Recall Curves** - For imbalanced datasets
4. **CV Boxplots** - Variability across folds
5. **ECE Comparison** - Bar chart with calibration errors
6. **Metrics Heatmap** - All metrics in a single figure
7. **Threshold Comparison** - With confidence intervals
8. **Statistical Significance Matrix** - p-values between methods

---

## Output Files

After running, the output folder contains:

```
outputs/calibration/
├── calibration_results_advanced.json    # All raw data
├── calibration_report_advanced.md       # Markdown report for the thesis
├── calibration_tables.tex               # LaTeX tables
├── calibrated_thresholds_advanced.py    # Python config for code
└── reliability_diagram_data.json        # Data for plotting
```

### Usage in Code

```python
from outputs.calibration.calibrated_thresholds_advanced import (
    CALIBRATED_THRESHOLDS,
    THRESHOLD_CONFIDENCE_INTERVALS,
    CALIBRATION_METADATA
)

# Use the calibrated threshold
ela_threshold = CALIBRATED_THRESHOLDS["ela"]  # 0.68

# Check confidence
ci = THRESHOLD_CONFIDENCE_INTERVALS["ela"]  # (0.62, 0.74)

# Get metadata
meta = CALIBRATION_METADATA["ela"]
print(f"AUC: {meta['auc']}, F1: {meta['f1']}")
```

---

## Dataset Structure for Calibration

```
dataset/
├── authentic/          # or: real/, original/, genuine/, 0/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── manipulated/        # or: fake/, tampered/, ai/, synthetic/, 1/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

Alternatively, with a labels file:

```json
// labels.json
{
    "image1.jpg": 0,
    "image2.jpg": 1,
    "image3.png": 0
}
```

```bash
python scripts/calibrate_thresholds_advanced.py \
    --dataset data/images \
    --labels data/labels.json
```

---

## Interpreting Results

### What constitutes a good result?

| Metric | Very Good | Acceptable | Poor |
|--------|-----------|------------|------|
| AUC-ROC | > 0.90 | 0.80-0.90 | < 0.80 |
| F1 | > 0.85 | 0.70-0.85 | < 0.70 |
| MCC | > 0.70 | 0.50-0.70 | < 0.50 |
| ECE | < 0.05 | 0.05-0.10 | > 0.10 |
| CV std | < 0.03 | 0.03-0.08 | > 0.08 |

### Statistical Significance

- **p < 0.05**: The difference is statistically significant
- **p < 0.01**: Highly significant difference
- **p >= 0.05**: We cannot reject the null hypothesis (methods are comparable)

---

## For the Thesis

### Sections to Include

1. **Methodology** (Section 4.X)
   - Present each forensic method
   - Include the bibliographic references from the output

2. **Experimental Results** (Section 5)
   - Copy the tables from `calibration_tables.tex`
   - Include the figures from `outputs/figures/`

3. **Discussion** (Section 6)
   - Analyze the confidence intervals
   - Compare with thresholds from the literature
   - Discuss the limitations of each method

### Example Paragraphs

```markdown
The optimal threshold for ELA was experimentally determined at 0.68
(95% CI: [0.62, 0.74]) by maximizing the F1 score on the CASIA v2.0 dataset.
This value is consistent with observations in the literature,
where Gunawan et al. (2017) report thresholds in the range of 0.6-0.8.
Cross-validation on 5 folds demonstrated good stability
(σ = 0.03), indicating robustness to data variation.
```

---

## Troubleshooting

### "No images found"
- Check the folder structure (it should be `authentic/` and `manipulated/`)
- Check the file extensions (.jpg, .jpeg, .png, .webp, .bmp)

### "Bootstrap failed"
- Increase the number of samples in the dataset (minimum 100)
- Use `--n-bootstrap 500` for small datasets

### "All threshold methods failed"
- Dataset too small or too imbalanced
- Ensure you have at least 20 images from each class

---

## References

1. **Youden's J Statistic**: Youden, W.J. (1950). Index for rating diagnostic tests.
2. **Temperature Scaling**: Guo, C. et al. (2017). On Calibration of Modern Neural Networks. ICML.
3. **McNemar Test**: McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages.
4. **DeLong Test**: DeLong, E.R. et al. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves.
5. **Bootstrap BCa**: Efron, B. (1987). Better bootstrap confidence intervals.

---

*Document generated for ImageTrust - Forensic Detection System.*
