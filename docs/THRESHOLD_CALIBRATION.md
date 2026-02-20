# Calibrarea Threshold-urilor - Ghid Metodologic

**Cerință profesor:** Fiecare threshold trebuie justificat teoretic și experimental, nu "pus de nebun".

---

## 1. Problema

Threshold-uri actuale fără justificare:
```python
# Exemple de valori NEJUSTIFICATE în cod:
ELA_THRESHOLD = 0.72        # De unde?
NOISE_THRESHOLD = 0.5       # De unde?
AI_CONFIDENCE_HIGH = 0.85   # De unde?
```

**Ce vrea profesorul:**
- Bază teoretică (articole)
- Validare experimentală (dataset)
- Capitol de comparație pentru fiecare metodă

---

## 2. Framework de Calibrare pentru Fiecare Metodă

### Structura pentru fiecare tehnic forensic:

```
1. TEORIE
   - Ce detectează metoda?
   - Articole de referință (minimum 3)
   - Ce spun articolele despre threshold?

2. EXPERIMENT
   - Dataset folosit (cu ground truth)
   - Metrici calculate (ROC curve, F1)
   - Threshold optim găsit

3. VALIDARE
   - Test pe dataset separat
   - Comparație cu threshold-uri din literatură
   - Analiza erorilor (false positives/negatives)
```

---

## 3. ELA (Error Level Analysis)

### 3.1 Teorie

**Ce detectează:**
ELA re-comprimă imaginea la un nivel JPEG cunoscut (ex: Q95) și compară cu originalul. Regiunile editate recent au erori diferite față de cele originale.

**Articole de referință:**

1. **Krawetz, N. (2007)** - "A Picture's Worth... Digital Image Analysis and Forensics"
   - Hack in the Box Conference
   - Introduce ELA pentru prima dată
   - NU dă threshold numeric, spune "look for inconsistencies"
   - Link: https://www.hackerfactor.com/papers/bh-usa-07-krawetz-wp.pdf

2. **Gunawan, T.S. et al. (2017)** - "Development of Photo Forensic using ELA"
   - International Journal of Electrical and Computer Engineering
   - Testează pe dataset de 200 imagini
   - Threshold variat 15-30 (în spațiu pixel, nu 0-1)
   - DOI: 10.11591/ijece.v7i5.pp2690-2695

3. **Sudiatmika, I.B. et al. (2019)** - "Image Forgery Detection Using ELA and CNN"
   - International Conference on Data Science and Information Technology
   - Combină ELA cu CNN
   - ELA ca preprocessing, nu threshold direct

**Concluzie teoretică:**
Articolele NU dau un threshold universal. Trebuie calibrat pe dataset.

### 3.2 Experiment - Cum calibrezi

```python
# Pseudocod pentru calibrare ELA threshold

import numpy as np
from sklearn.metrics import roc_curve, f1_score

def calibrate_ela_threshold(dataset_path, ground_truth):
    """
    dataset_path: folder cu imagini
    ground_truth: dict {image_name: 0=authentic, 1=manipulated}
    """

    scores = []
    labels = []

    for image_path in dataset_path:
        # Calculează ELA score (normalizat 0-1)
        ela_score = compute_ela_score(image_path)
        scores.append(ela_score)
        labels.append(ground_truth[image_path.name])

    scores = np.array(scores)
    labels = np.array(labels)

    # Găsește threshold optim
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Metoda 1: Youden's J statistic (maximizează TPR - FPR)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold_youden = thresholds[optimal_idx]

    # Metoda 2: Maximizează F1
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

### 3.3 Dataset-uri pentru validare ELA

| Dataset | Imagini | Tip | Link |
|---------|---------|-----|------|
| CASIA v2.0 | 12,614 | Splicing, Copy-Move | http://forensics.idealtest.org |
| Columbia | 1,845 | Splicing | https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/ |
| COVERAGE | 100 pairs | Copy-Move | https://github.com/wenbihan/coverage |
| IMD2020 | 35,000 | Mixed manipulation | https://staff.fnwi.uva.nl/z.geradts/forensicimagetools/ |

### 3.4 Rezultate Așteptate (Exemplu)

```
Dataset: CASIA v2.0
Imagini testate: 5,123
Autentice: 2,500 | Manipulate: 2,623

Rezultate calibrare:
- Threshold optim (Youden): 0.68
- Threshold optim (F1): 0.71
- AUC: 0.823
- F1 la threshold 0.71: 0.79

Concluzie: Threshold ELA = 0.70 (rotunjit)
Justificare: Optimizat pentru F1 pe CASIA v2.0
```

---

## 4. Noise Inconsistency Analysis

### 4.1 Teorie

**Ce detectează:**
Camerele digitale introduc noise caracteristic. Regiunile din surse diferite au pattern-uri de noise diferite.

**Articole de referință:**

1. **Mahdian, B. & Saic, S. (2009)** - "Using noise inconsistencies for blind image forensics"
   - Image and Vision Computing, Vol 27
   - Propune analiza varianței noise-ului pe blocuri
   - DOI: 10.1016/j.imavis.2008.06.010

2. **Pan, X. et al. (2011)** - "Region Duplication Detection Using Image Feature Matching"
   - IEEE Transactions on Information Forensics and Security
   - Noise + features pentru copy-move

3. **Cozzolino, D. et al. (2015)** - "Efficient dense-field copy–move forgery detection"
   - IEEE Transactions on Information Forensics and Security
   - State-of-art noise analysis

### 4.2 Calibrare

Similar cu ELA:
1. Dataset cu ground truth (regiuni manipulate marcate)
2. Calculează noise variance per bloc
3. Găsește threshold pentru "inconsistență semnificativă"
4. Validează pe dataset separat

---

## 5. JPEG Artifacts (Double JPEG Detection)

### 5.1 Teorie

**Ce detectează:**
Prima compresie JPEG lasă "fantome" în histograma coeficienților DCT. Re-compresia creează pattern periodic detectabil.

**Articole de referință:**

1. **Farid, H. (2009)** - "Exposing Digital Forgeries from JPEG Ghosts"
   - IEEE Transactions on Information Forensics and Security
   - Introduce "JPEG ghosts"
   - DOI: 10.1109/TIFS.2008.2012215

2. **Bianchi, T. & Piva, A. (2012)** - "Image Forgery Localization via Block-Grained Analysis"
   - IEEE Transactions on Information Forensics and Security
   - Analiză DCT per bloc

3. **Li, W. et al. (2015)** - "Passive Detection of Doctored JPEG Image via Block Artifact Grid Extraction"
   - Signal Processing
   - Grid artifacts detection

### 5.2 Calibrare

```python
def calibrate_jpeg_threshold(dataset):
    """
    Pentru double-JPEG detection:
    - 0 = single compression (sau PNG convertit)
    - 1 = multiple compressions detected
    """
    # Calcul bazat pe periodicitate în histograma DCT
    # Threshold = cât de "puternică" e periodicitatea
```

---

## 6. AI Detection (HuggingFace Models)

### 6.1 Teorie

**Ce detectează:**
Modelele CNN/ViT antrenate pe perechi real/AI învață pattern-uri specifice imaginilor sintetice.

**Articole pentru modele folosite:**

1. **umm-maybe/AI-image-detector**
   - Bazat pe: Wang et al. "CNN-generated images are surprisingly easy to spot... for now"
   - CVPR 2020
   - https://arxiv.org/abs/1912.11035

2. **Organika/sdxl-detector**
   - Specific pentru Stable Diffusion XL
   - Fine-tuned pe SDXL outputs

3. **Corvi et al. (2023)** - "On The Detection of Synthetic Images Generated by Diffusion Models"
   - ICASSP 2023
   - Analizează detectabilitatea diffusion models

### 6.2 Calibrare

Modelele HF returnează deja probabilități. Dar trebuie:
1. Verificat dacă sunt calibrate (Expected Calibration Error)
2. Aplicat Temperature Scaling dacă nu
3. Setat threshold pentru HIGH/MEDIUM/LOW confidence

```python
def calibrate_ai_detector_threshold(model, dataset):
    """
    Dataset cu imagini AI și reale, etichetate
    """
    probs = model.predict_proba(dataset.images)
    labels = dataset.labels  # 0=real, 1=AI

    # Calculează ECE (Expected Calibration Error)
    ece = compute_ece(probs, labels, n_bins=10)

    if ece > 0.05:  # Prost calibrat
        # Aplică temperature scaling
        temperature = find_optimal_temperature(probs, labels)
        calibrated_probs = softmax(logits / temperature)

    # Găsește threshold-uri pentru confidence levels
    # HIGH = minimizează false positives (nu acuzăm pe nedrept)
    # LOW = minimizează false negatives (nu ratăm AI-uri)
```

---

## 7. Structura Capitolului de Comparație (Teză)

Pentru FIECARE metodă, scrii:

```markdown
## 4.X [Numele Metodei] - Error Level Analysis

### 4.X.1 Fundament Teoretic
- Ce principiu fizic/matematic stă la bază
- Când a fost propusă, de cine
- Ecuații relevante

### 4.X.2 Articole de Referință
- [Articol 1] - ce spune
- [Articol 2] - ce spune
- [Articol 3] - ce spune
- Threshold-uri raportate în literatură (dacă există)

### 4.X.3 Implementare în ImageTrust
- Algoritm folosit
- Parametri configurabili
- Diferențe față de implementări standard

### 4.X.4 Calibrare Experimentală
- Dataset folosit: [nume, dimensiune, sursă]
- Metrici calculate: [AUC, F1, precision, recall]
- Threshold optim găsit: X.XX
- Metodă de selecție: [Youden's J / Max F1 / altă metodă]

### 4.X.5 Rezultate și Discuție
- Tabel cu rezultate
- Curba ROC
- Comparație cu threshold-uri din literatură
- Limitări identificate

### 4.X.6 Concluzie
- Threshold final adoptat: X.XX
- Justificare în 2-3 propoziții
```

---

## 8. Dataset-uri Recomandate pentru Calibrare

### Pentru Manipulare Tradițională (ELA, Noise, JPEG):

| Dataset | N | Tip | Download |
|---------|---|-----|----------|
| **CASIA v1.0** | 1,725 | Splicing | forensics.idealtest.org |
| **CASIA v2.0** | 12,614 | Splicing + CM | forensics.idealtest.org |
| **Columbia** | 1,845 | Splicing | columbia.edu |
| **COVERAGE** | 100 | Copy-Move | github |
| **CoMoFoD** | 10,000 | Copy-Move | comofod.org |
| **GRIP** | 80 | Retouching | github |

### Pentru AI Detection:

| Dataset | N | Generatori | Download |
|---------|---|------------|----------|
| **GenImage** | 1.3M | SD, Midjourney, DALL-E, etc. | github.com/GenImage-Dataset |
| **CIFAKE** | 120K | SD 1.4 | kaggle |
| **AI-ArtBench** | 180K | Multiple | huggingface |
| **DiffusionDB** | 14M | SD | poloclub.github.io/diffusiondb |

---

## 9. Checklist Final

Pentru fiecare metodă (ELA, Noise, JPEG, AI Detection, etc.):

- [ ] Am găsit minimum 3 articole de referință
- [ ] Am ales un dataset cu ground truth
- [ ] Am rulat calibrarea și am obținut threshold optim
- [ ] Am calculat AUC și F1 la acel threshold
- [ ] Am validat pe un dataset separat (nu același cu calibrarea)
- [ ] Am scris secțiunea în teză cu toate justificările
- [ ] Am comparat threshold-ul meu cu ce spune literatura

---

## 10. Exemplu de Tabel Final pentru Teză

| Metodă | Threshold | Sursă Threshold | AUC | F1 | Dataset Calibrare |
|--------|-----------|-----------------|-----|----|--------------------|
| ELA | 0.70 | Calibrat experimental | 0.82 | 0.79 | CASIA v2.0 |
| Noise | 0.55 | Calibrat experimental | 0.75 | 0.71 | CASIA v2.0 |
| JPEG Double | 0.60 | Farid (2009) + calibrare | 0.88 | 0.84 | Columbia |
| AI Detection | 0.85 (HIGH) | Calibrat + Temperature | 0.94 | 0.91 | GenImage |

**Notă:** Fiecare valoare are justificare în secțiunea corespunzătoare.

---

*Document creat pentru a răspunde cerințelor profesorului privind justificarea threshold-urilor.*
