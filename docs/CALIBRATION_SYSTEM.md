# Sistemul de Calibrare a Threshold-urilor

## Prezentare Generală

Sistemul de calibrare implementat în ImageTrust oferă validare statistică riguroasă pentru toate threshold-urile utilizate în metodele forensic, conform cerințelor academice pentru teză de master.

**Cerința profesorului:** "Nu pune valori de nebun, demonstrează-le experimental!"

---

## Fișiere Principale

| Fișier | Descriere |
|--------|-----------|
| `scripts/calibrate_thresholds.py` | Script de calibrare de bază |
| `scripts/calibrate_thresholds_advanced.py` | **Script avansat** cu toate funcționalitățile |
| `scripts/generate_calibration_plots.py` | Generare figuri pentru teză |
| `docs/THRESHOLD_CALIBRATION.md` | Ghid metodologic |

---

## Funcționalități Avansate

### 1. Metode de Selecție Threshold

| Metodă | Descriere | Când să folosești |
|--------|-----------|-------------------|
| **Youden's J** | Maximizează TPR - FPR | Dataset balansat |
| **F1 Max** | Maximizează F1 score | Dataset moderat dezechilibrat |
| **F2 Max** | Favorizează recall | Când false negatives sunt costisitoare |
| **F0.5 Max** | Favorizează precision | Când false positives sunt costisitoare |
| **MCC Max** | Maximizează Matthews Correlation | Dataset foarte dezechilibrat |
| **G-Mean** | √(TPR × TNR) | Clasă minoritară importantă |
| **EER** | FPR = FNR | Aplicații biometrice |
| **Cost-Sensitive** | Minimizează costul ponderat | Costuri diferite pentru erori |

### 2. Cross-Validation

```bash
python scripts/calibrate_thresholds_advanced.py \
    --dataset data/calibration \
    --k-folds 5 \
    --output outputs/calibration
```

- Stratified K-Fold (păstrează proporția claselor)
- Threshold-ul se găsește pe training, se evaluează pe validation
- Raportează mean ± std pentru toate metricile

### 3. Bootstrap Confidence Intervals

- **BCa (Bias-Corrected and Accelerated)**: Mai robust decât percentile simplu
- Intervale la 95% și 99%
- Implicit: 1000 resample-uri

```python
# Exemplu rezultat
threshold_optimal: 0.68
confidence_interval_95: [0.62, 0.74]
confidence_interval_99: [0.59, 0.77]
cv_threshold_std: 0.03
```

### 4. Calibrarea Probabilităților

#### Temperature Scaling
```
P_calibrated = sigmoid(logits / T)
```
- Optimizează T pentru a minimiza Negative Log Likelihood
- Tipic: T ∈ [0.5, 2.0]

#### Metrici de Calibrare
- **ECE (Expected Calibration Error)**: Eroare medie de calibrare
- **MCE (Maximum Calibration Error)**: Worst-case calibration
- **Brier Score**: Mean squared error pentru probabilități

### 5. Teste Statistice

#### McNemar Test
- Compară două clasificatoare pe același dataset
- Testează dacă diferența în erori este semnificativă

#### DeLong Test
- Compară AUC-urile a două modele
- Include interval de încredere pentru diferența de AUC

#### Wilcoxon Signed-Rank
- Compară performanța pe folds de cross-validation
- Non-parametric, nu presupune normalitate

---

## Utilizare

### Calibrare Completă

```bash
# Calibrare pentru toate metodele
python scripts/calibrate_thresholds_advanced.py \
    --dataset data/casia_v2 \
    --dataset-name "CASIA v2.0" \
    --output outputs/calibration \
    --k-folds 5 \
    --n-bootstrap 1000 \
    --seed 42

# Doar o singură metodă
python scripts/calibrate_thresholds_advanced.py \
    --dataset data/casia_v2 \
    --method ela \
    --output outputs/calibration_ela
```

### Generare Figuri

```bash
python scripts/generate_calibration_plots.py \
    --results outputs/calibration \
    --output outputs/figures \
    --format pdf
```

### Figuri Generate

1. **Reliability Diagrams** - Curbe de calibrare
2. **ROC Curves** - Comparație între metode
3. **Precision-Recall Curves** - Pentru dataset dezechilibrat
4. **CV Boxplots** - Variabilitate între folds
5. **ECE Comparison** - Bar chart cu erori de calibrare
6. **Metrics Heatmap** - Toate metricile într-o figură
7. **Threshold Comparison** - Cu intervale de încredere
8. **Statistical Significance Matrix** - p-values între metode

---

## Output Files

După rulare, în folderul de output găsești:

```
outputs/calibration/
├── calibration_results_advanced.json    # Toate datele raw
├── calibration_report_advanced.md       # Raport Markdown pentru teză
├── calibration_tables.tex               # Tabele LaTeX
├── calibrated_thresholds_advanced.py    # Config Python pentru cod
└── reliability_diagram_data.json        # Date pentru plotare
```

### Utilizare în Cod

```python
from outputs.calibration.calibrated_thresholds_advanced import (
    CALIBRATED_THRESHOLDS,
    THRESHOLD_CONFIDENCE_INTERVALS,
    CALIBRATION_METADATA
)

# Folosește threshold-ul calibrat
ela_threshold = CALIBRATED_THRESHOLDS["ela"]  # 0.68

# Verifică încrederea
ci = THRESHOLD_CONFIDENCE_INTERVALS["ela"]  # (0.62, 0.74)

# Obține metadata
meta = CALIBRATION_METADATA["ela"]
print(f"AUC: {meta['auc']}, F1: {meta['f1']}")
```

---

## Structura Dataset pentru Calibrare

```
dataset/
├── authentic/          # sau: real/, original/, genuine/, 0/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── manipulated/        # sau: fake/, tampered/, ai/, synthetic/, 1/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

Alternativ, cu fișier de etichete:

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

## Interpretare Rezultate

### Ce înseamnă un rezultat bun?

| Metrică | Foarte Bun | Acceptabil | Slab |
|---------|------------|------------|------|
| AUC-ROC | > 0.90 | 0.80-0.90 | < 0.80 |
| F1 | > 0.85 | 0.70-0.85 | < 0.70 |
| MCC | > 0.70 | 0.50-0.70 | < 0.50 |
| ECE | < 0.05 | 0.05-0.10 | > 0.10 |
| CV std | < 0.03 | 0.03-0.08 | > 0.08 |

### Semnificație Statistică

- **p < 0.05**: Diferența este semnificativă statistic
- **p < 0.01**: Diferență foarte semnificativă
- **p ≥ 0.05**: Nu putem respinge ipoteza nulă (metodele sunt comparabile)

---

## Pentru Teză

### Secțiuni de Inclus

1. **Metodologie** (Section 4.X)
   - Prezintă fiecare metodă forensic
   - Include referințele bibliografice din output

2. **Rezultate Experimentale** (Section 5)
   - Copiază tabelele din `calibration_tables.tex`
   - Include figurile din `outputs/figures/`

3. **Discuție** (Section 6)
   - Analizează intervalele de încredere
   - Compară cu threshold-uri din literatură
   - Discută limitările fiecărei metode

### Exemplu de Paragrafe

```markdown
Threshold-ul optim pentru ELA a fost determinat experimental la 0.68
(95% CI: [0.62, 0.74]) prin maximizarea F1 score pe dataset-ul CASIA v2.0.
Valoarea este consistentă cu observațiile din literatura de specialitate,
unde Gunawan et al. (2017) raportează threshold-uri în intervalul 0.6-0.8.
Cross-validarea pe 5 folds a demonstrat stabilitate bună
(σ = 0.03), indicând robustețe la variația datelor.
```

---

## Troubleshooting

### "No images found"
- Verifică structura folderelor (trebuie să fie `authentic/` și `manipulated/`)
- Verifică extensiile fișierelor (.jpg, .jpeg, .png, .webp, .bmp)

### "Bootstrap failed"
- Crește numărul de eșantioane în dataset (minim 100)
- Folosește `--n-bootstrap 500` pentru dataset-uri mici

### "All threshold methods failed"
- Dataset prea mic sau prea dezechilibrat
- Verifică că ai cel puțin 20 imagini din fiecare clasă

---

## Referințe

1. **Youden's J Statistic**: Youden, W.J. (1950). Index for rating diagnostic tests.
2. **Temperature Scaling**: Guo, C. et al. (2017). On Calibration of Modern Neural Networks. ICML.
3. **McNemar Test**: McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages.
4. **DeLong Test**: DeLong, E.R. et al. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves.
5. **Bootstrap BCa**: Efron, B. (1987). Better bootstrap confidence intervals.

---

*Document generat pentru ImageTrust - Sistem de Detecție Forensic.*
