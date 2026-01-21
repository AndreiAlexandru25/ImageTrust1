# 📊 Performance Comparison (Baselines vs ImageTrust)

This section provides a structured template for comparing ImageTrust against existing methods, as requested for the dissertation. It is designed to be reproducible using the included evaluation scripts.

---

## 1) Evaluation Setup

### Datasets
Use the following dataset splits (examples):
- **Real**: phone camera photos (incl. WhatsApp/Instagram compressed)
- **AI**: DALL‑E 2/3, Midjourney v4/v5/v6, SD 1.5/2.1/XL
- **Manipulated**: Photoshop/Canva edits (splicing/copy‑move)

Recommended structure:
```
data/eval/
  real/
  ai/
  manipulated/
```

### Metrics
Report at least:
- Accuracy, Precision, Recall, F1
- ROC‑AUC
- ECE (calibration)
- Robustness under JPEG/blur (optional)

---

## 2) Baseline Methods

Compare against:
- Single model (best individual HF detector)
- Ensemble (ImageTrust ML only)
- Ensemble + calibration (WhatsApp/Instagram)
- External services (Illuminarty / Hive demo, optional)

---

## 3) Results Table (Template)

| Method | Accuracy | F1 | ROC‑AUC | ECE ↓ | Notes |
|--------|----------|----|--------|------|------|
| Single Model (Best) | — | — | — | — | baseline |
| ImageTrust ML‑Only | — | — | — | — | ensemble |
| ImageTrust + Calibration | — | — | — | — | WhatsApp/Instagram |
| External (Optional) | — | — | — | — | reference |

---

## 4) Improvement Analysis

Summarize the gains:
- **Δ Accuracy** from single model → ensemble
- **Δ ECE** with calibration
- **Robustness** improvements after compression

Example narrative:
> “Compared with the best single HF detector, the ImageTrust ensemble improved F1 by +X% and reduced calibration error by Y. WhatsApp calibration removed Z% false positives on compressed images.”

---

## 5) How to Run Evaluation

Use existing evaluation script:
```
python scripts/run_evaluation.py --dataset data/eval
python scripts/run_evaluation.py --cross-generator
python scripts/run_evaluation.py --ablation
```

Results are stored in `outputs/` for reproducibility.

