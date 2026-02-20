# Modulul de Detecție Steganografică

## Prezentare Generală

Modulul de steganografie implementează 7 metode de analiză pentru detectarea mesajelor ascunse în imagini. Este integrat în sistemul de forensics al ImageTrust.

**Locație:** `src/imagetrust/forensics/steganography.py`

---

## Metode Implementate

### 1. LSB Analysis (Least Significant Bit)

**Principiu:** Analizează distribuția biților cel mai puțin semnificativi.

```python
from imagetrust.forensics.steganography import LSBAnalyzer

analyzer = LSBAnalyzer()
result = analyzer.analyze(image_array)
# result.detection_probability: 0.0 - 1.0
# result.anomalies: List[str]
```

**Ce detectează:**
- Mesaje ascunse în LSB al pixelilor
- Pattern-uri nenaturale în distribuția LSB
- Steganografie naivă (sequential embedding)

**Referințe:**
- Westfeld & Pfitzmann (1999): "Attacks on Steganographic Systems"
- Dumitrescu et al. (2003): "Detection of LSB Steganography"

---

### 2. Chi-Square Attack

**Principiu:** Detectează perechi de valori cu frecvențe anormale.

```python
from imagetrust.forensics.steganography import ChiSquareAnalyzer

analyzer = ChiSquareAnalyzer()
result = analyzer.analyze(image_array)
# result.chi_square_statistic
# result.p_value
# result.is_suspicious (p < 0.05)
```

**Teorie:**
- LSB embedding creează perechi "Pairs of Values" (PoV)
- 2k și 2k+1 devin echiprobabile după embedding
- Test χ² detectează această uniformizare

**Sensibilitate:** Detectează embedding de ~10-15% din capacitate.

---

### 3. RS Analysis (Regular-Singular)

**Principiu:** Analizează grupuri de pixeli bazat pe funcții de discriminare.

```python
from imagetrust.forensics.steganography import RSAnalyzer

analyzer = RSAnalyzer()
result = analyzer.analyze(image_array)
# result.estimated_message_length: float (0.0 - 1.0)
# result.r_m, result.s_m: Regular/Singular groups
```

**Grupuri:**
- **Regular (R):** Netezimea crește după flipping LSB
- **Singular (S):** Netezimea scade după flipping
- **Unusable (U):** Netezime neschimbată

**Estimare:** Lungimea mesajului se estimează din relația:
```
R_M - S_M ≈ R_{-M} - S_{-M}  (pentru imagine originală)
```

**Precizie:** ±5% pentru mesaje > 20% din capacitate.

---

### 4. Sample Pair Analysis (SPA)

**Principiu:** Extindere a RS, analizează perechi de pixeli adiacenți.

```python
from imagetrust.forensics.steganography import SPAAnalyzer

analyzer = SPAAnalyzer()
result = analyzer.analyze(image_array)
# result.p_value
# result.estimated_embedding_rate
```

**Avantaj:** Mai precis decât RS pentru embedding rate mic (<10%).

---

### 5. Histogram Analysis

**Principiu:** Detectează anomalii în histograma valorilor de pixeli.

```python
from imagetrust.forensics.steganography import HistogramAnalyzer

analyzer = HistogramAnalyzer()
result = analyzer.analyze(image_array)
# result.histogram_anomalies
# result.smoothness_score
# result.peak_analysis
```

**Ce detectează:**
- "Combing effect" (alternare high-low în bins adiacente)
- Netezime anormală a histogramei
- Peaks/valleys nenaturale

---

### 6. DCT Analysis (pentru JPEG)

**Principiu:** Analizează coeficienții DCT pentru JPEG steganografie.

```python
from imagetrust.forensics.steganography import DCTAnalyzer

analyzer = DCTAnalyzer()
result = analyzer.analyze(image_path)  # Necesită calea fișierului
# result.dct_histogram_anomalies
# result.zero_coefficient_ratio
# result.jsteg_indicator
```

**Detectează:**
- JSteg embedding (modificarea coeficienților non-zero)
- F5 traces (matrix embedding)
- OutGuess artifacts

**Notă:** Funcționează doar cu imagini JPEG (nu PNG/BMP).

---

### 7. Visual Attack (Bit Plane Visualization)

**Principiu:** Generează vizualizări ale planurilor de biți.

```python
from imagetrust.forensics.steganography import VisualAttackAnalyzer

analyzer = VisualAttackAnalyzer()
visualizations = analyzer.generate_bit_planes(image_array)
# visualizations: Dict[int, np.ndarray]  # plane 0-7

# Detectare automată pattern-uri
result = analyzer.detect_visual_patterns(image_array)
# result.suspicious_planes: List[int]
# result.pattern_descriptions: List[str]
```

**Utilizare:** Pentru inspecție manuală de către expert.

---

## Utilizare Combinată

### SteganographyDetector (All-in-One)

```python
from imagetrust.forensics.steganography import SteganographyDetector

detector = SteganographyDetector()

# Analiză completă
report = detector.analyze(image_path)

print(f"Stego probability: {report.overall_probability:.2%}")
print(f"Confidence: {report.confidence}")
print(f"Methods triggered: {[m.value for m in report.detected_methods]}")

# Detalii per metodă
for method, result in report.method_results.items():
    print(f"{method}: {result.detection_probability:.3f}")
```

### Raport Detaliat

```python
# Generare raport text
text_report = detector.generate_report(report)
print(text_report)

# Sau ca dicționar pentru JSON
dict_report = report.to_dict()
```

---

## Integrare cu ForensicsPlugin

```python
from imagetrust.forensics.steganography import create_steganography_plugin

# Crează plugin pentru sistemul de forensics
plugin = create_steganography_plugin()

# Folosire în pipeline-ul forensic existent
from imagetrust.forensics import ForensicsEngine

engine = ForensicsEngine()
engine.register_plugin(plugin)

# Acum steganografia e inclusă în analiza completă
results = engine.analyze(image_path)
```

---

## Configurare

### Thresholds

```python
from imagetrust.forensics.steganography import (
    SteganographyDetector,
    StegConfidence
)

detector = SteganographyDetector(
    lsb_threshold=0.6,        # Probabilitate LSB suspicioasă
    chi_square_alpha=0.05,    # Nivel de semnificație χ²
    rs_threshold=0.1,         # RS estimated message length
    confidence_weights={
        "lsb": 0.25,
        "chi_square": 0.20,
        "rs": 0.20,
        "spa": 0.15,
        "histogram": 0.10,
        "dct": 0.10,
    }
)
```

### Selectare Metode

```python
# Doar anumite metode
detector = SteganographyDetector(
    enabled_methods=["lsb", "chi_square", "rs"]
)

# Exclude metode lente
detector = SteganographyDetector(
    exclude_methods=["visual_attack"]  # Vizualizarea e lentă
)
```

---

## CLI Usage

```bash
# Analiză imagine singulară
python -m imagetrust.forensics.steganography image.png

# Cu output detaliat
python -m imagetrust.forensics.steganography image.png --verbose

# Salvare raport
python -m imagetrust.forensics.steganography image.png --output report.json

# Analiză folder
python -m imagetrust.forensics.steganography data/images/ --batch
```

---

## Interpretare Rezultate

### Niveluri de Încredere

| Probabilitate | Încredere | Interpretare |
|--------------|-----------|--------------|
| 0.0 - 0.3 | LOW | Probabil curat |
| 0.3 - 0.6 | MEDIUM | Necesită investigare |
| 0.6 - 0.8 | HIGH | Probabil conține date ascunse |
| 0.8 - 1.0 | VERY_HIGH | Aproape sigur steganografie |

### Semnale de Alertă

1. **Multiple metode trigger**: Dacă 3+ metode dau score > 0.5, e suspect
2. **Chi-square p < 0.01**: Foarte probabil modificat
3. **RS estimate > 0.15**: Message length semnificativ
4. **LSB entropy anormală**: Pattern nenatural în biți

---

## Limitări

### Ce NU detectează bine:

1. **Steganografie avansată:**
   - Steghide (embedding în DCT cu parolă)
   - OutGuess (preservă statistici first-order)
   - F5 cu matrix embedding

2. **Embedding foarte mic:**
   - Sub 5% din capacitate, greu de detectat
   - False negatives mai probabile

3. **Imagini procesate:**
   - După resize/compress, traces se distrug
   - JPEG re-compression elimină multe artifacts

### False Positives frecvente:

- Imagini cu texturi foarte regulate (grids, patterns)
- Imagini generate AI (distribuții nenaturale)
- Screenshots cu text/UI elements

---

## Referințe Bibliografice

1. **Westfeld, A. & Pfitzmann, A.** (1999). Attacks on Steganographic Systems. Information Hiding.

2. **Fridrich, J. et al.** (2001). Reliable Detection of LSB Steganography. ACM Workshop on Multimedia and Security.

3. **Dumitrescu, S. et al.** (2003). Detection of LSB Steganography via Sample Pair Analysis. IEEE Transactions on Signal Processing.

4. **Fridrich, J. & Goljan, M.** (2002). Practical Steganalysis of Digital Images. SPIE Electronic Imaging.

5. **Ker, A.D.** (2005). Steganalysis of LSB matching in grayscale images. IEEE Signal Processing Letters.

---

## Exemplu Complet

```python
#!/usr/bin/env python3
"""Exemplu complet de analiză steganografică."""

from pathlib import Path
from imagetrust.forensics.steganography import SteganographyDetector

def analyze_image(image_path: str):
    detector = SteganographyDetector()
    report = detector.analyze(Path(image_path))

    print(f"=== Analiză Steganografică: {image_path} ===\n")
    print(f"Probabilitate generală: {report.overall_probability:.1%}")
    print(f"Nivel de încredere: {report.confidence.value}")
    print(f"Verdict: {'SUSPECT' if report.overall_probability > 0.5 else 'PROBABIL CURAT'}")

    print("\n--- Rezultate per metodă ---")
    for method, result in report.method_results.items():
        status = "⚠️" if result.detection_probability > 0.5 else "✓"
        print(f"  {status} {method}: {result.detection_probability:.3f}")
        if result.anomalies:
            for anomaly in result.anomalies[:3]:
                print(f"      - {anomaly}")

    if report.detected_methods:
        print(f"\n⚠️  Metode care au detectat anomalii: {[m.value for m in report.detected_methods]}")

    return report

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_image(sys.argv[1])
    else:
        print("Usage: python example.py <image_path>")
```

---

*Document generat pentru ImageTrust - Sistem de Detecție Forensic.*
