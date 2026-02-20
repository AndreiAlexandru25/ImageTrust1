# Steganographic Detection Module

## Overview

The steganography module implements 7 analysis methods for detecting hidden messages in images. It is integrated into the ImageTrust forensics system.

**Location:** `src/imagetrust/forensics/steganography.py`

---

## Implemented Methods

### 1. LSB Analysis (Least Significant Bit)

**Principle:** Analyzes the distribution of least significant bits.

```python
from imagetrust.forensics.steganography import LSBAnalyzer

analyzer = LSBAnalyzer()
result = analyzer.analyze(image_array)
# result.detection_probability: 0.0 - 1.0
# result.anomalies: List[str]
```

**What it detects:**
- Messages hidden in the LSB of pixels
- Unnatural patterns in LSB distribution
- Naive steganography (sequential embedding)

**References:**
- Westfeld & Pfitzmann (1999): "Attacks on Steganographic Systems"
- Dumitrescu et al. (2003): "Detection of LSB Steganography"

---

### 2. Chi-Square Attack

**Principle:** Detects pairs of values with abnormal frequencies.

```python
from imagetrust.forensics.steganography import ChiSquareAnalyzer

analyzer = ChiSquareAnalyzer()
result = analyzer.analyze(image_array)
# result.chi_square_statistic
# result.p_value
# result.is_suspicious (p < 0.05)
```

**Theory:**
- LSB embedding creates "Pairs of Values" (PoV)
- 2k and 2k+1 become equiprobable after embedding
- The chi-square test detects this uniformization

**Sensitivity:** Detects embedding of ~10-15% of capacity.

---

### 3. RS Analysis (Regular-Singular)

**Principle:** Analyzes pixel groups based on discrimination functions.

```python
from imagetrust.forensics.steganography import RSAnalyzer

analyzer = RSAnalyzer()
result = analyzer.analyze(image_array)
# result.estimated_message_length: float (0.0 - 1.0)
# result.r_m, result.s_m: Regular/Singular groups
```

**Groups:**
- **Regular (R):** Smoothness increases after LSB flipping
- **Singular (S):** Smoothness decreases after flipping
- **Unusable (U):** Smoothness unchanged

**Estimation:** The message length is estimated from the relationship:
```
R_M - S_M ≈ R_{-M} - S_{-M}  (for the original image)
```

**Accuracy:** ±5% for messages > 20% of capacity.

---

### 4. Sample Pair Analysis (SPA)

**Principle:** Extension of RS, analyzes pairs of adjacent pixels.

```python
from imagetrust.forensics.steganography import SPAAnalyzer

analyzer = SPAAnalyzer()
result = analyzer.analyze(image_array)
# result.p_value
# result.estimated_embedding_rate
```

**Advantage:** More accurate than RS for low embedding rates (<10%).

---

### 5. Histogram Analysis

**Principle:** Detects anomalies in the pixel value histogram.

```python
from imagetrust.forensics.steganography import HistogramAnalyzer

analyzer = HistogramAnalyzer()
result = analyzer.analyze(image_array)
# result.histogram_anomalies
# result.smoothness_score
# result.peak_analysis
```

**What it detects:**
- "Combing effect" (alternating high-low in adjacent bins)
- Abnormal histogram smoothness
- Unnatural peaks/valleys

---

### 6. DCT Analysis (for JPEG)

**Principle:** Analyzes DCT coefficients for JPEG steganography.

```python
from imagetrust.forensics.steganography import DCTAnalyzer

analyzer = DCTAnalyzer()
result = analyzer.analyze(image_path)  # Requires the file path
# result.dct_histogram_anomalies
# result.zero_coefficient_ratio
# result.jsteg_indicator
```

**Detects:**
- JSteg embedding (modification of non-zero coefficients)
- F5 traces (matrix embedding)
- OutGuess artifacts

**Note:** Works only with JPEG images (not PNG/BMP).

---

### 7. Visual Attack (Bit Plane Visualization)

**Principle:** Generates visualizations of bit planes.

```python
from imagetrust.forensics.steganography import VisualAttackAnalyzer

analyzer = VisualAttackAnalyzer()
visualizations = analyzer.generate_bit_planes(image_array)
# visualizations: Dict[int, np.ndarray]  # plane 0-7

# Automatic pattern detection
result = analyzer.detect_visual_patterns(image_array)
# result.suspicious_planes: List[int]
# result.pattern_descriptions: List[str]
```

**Usage:** For manual inspection by an expert.

---

## Combined Usage

### SteganographyDetector (All-in-One)

```python
from imagetrust.forensics.steganography import SteganographyDetector

detector = SteganographyDetector()

# Full analysis
report = detector.analyze(image_path)

print(f"Stego probability: {report.overall_probability:.2%}")
print(f"Confidence: {report.confidence}")
print(f"Methods triggered: {[m.value for m in report.detected_methods]}")

# Details per method
for method, result in report.method_results.items():
    print(f"{method}: {result.detection_probability:.3f}")
```

### Detailed Report

```python
# Generate text report
text_report = detector.generate_report(report)
print(text_report)

# Or as a dictionary for JSON
dict_report = report.to_dict()
```

---

## Integration with ForensicsPlugin

```python
from imagetrust.forensics.steganography import create_steganography_plugin

# Create plugin for the forensics system
plugin = create_steganography_plugin()

# Use in the existing forensic pipeline
from imagetrust.forensics import ForensicsEngine

engine = ForensicsEngine()
engine.register_plugin(plugin)

# Now steganography is included in the full analysis
results = engine.analyze(image_path)
```

---

## Configuration

### Thresholds

```python
from imagetrust.forensics.steganography import (
    SteganographyDetector,
    StegConfidence
)

detector = SteganographyDetector(
    lsb_threshold=0.6,        # Suspicious LSB probability
    chi_square_alpha=0.05,    # Chi-square significance level
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

### Method Selection

```python
# Only specific methods
detector = SteganographyDetector(
    enabled_methods=["lsb", "chi_square", "rs"]
)

# Exclude slow methods
detector = SteganographyDetector(
    exclude_methods=["visual_attack"]  # Visualization is slow
)
```

---

## CLI Usage

```bash
# Analyze a single image
python -m imagetrust.forensics.steganography image.png

# With verbose output
python -m imagetrust.forensics.steganography image.png --verbose

# Save report
python -m imagetrust.forensics.steganography image.png --output report.json

# Analyze a folder
python -m imagetrust.forensics.steganography data/images/ --batch
```

---

## Interpreting Results

### Confidence Levels

| Probability | Confidence | Interpretation |
|-------------|------------|----------------|
| 0.0 - 0.3 | LOW | Probably clean |
| 0.3 - 0.6 | MEDIUM | Requires investigation |
| 0.6 - 0.8 | HIGH | Likely contains hidden data |
| 0.8 - 1.0 | VERY_HIGH | Almost certainly steganography |

### Alert Signals

1. **Multiple methods triggered**: If 3+ methods give a score > 0.5, it is suspicious
2. **Chi-square p < 0.01**: Very likely modified
3. **RS estimate > 0.15**: Significant message length
4. **Abnormal LSB entropy**: Unnatural pattern in bits

---

## Limitations

### What it does NOT detect well:

1. **Advanced steganography:**
   - Steghide (embedding in DCT with password)
   - OutGuess (preserves first-order statistics)
   - F5 with matrix embedding

2. **Very small embedding:**
   - Below 5% of capacity, hard to detect
   - False negatives more likely

3. **Processed images:**
   - After resize/compress, traces are destroyed
   - JPEG re-compression eliminates many artifacts

### Common false positives:

- Images with very regular textures (grids, patterns)
- AI-generated images (unnatural distributions)
- Screenshots with text/UI elements

---

## Bibliographic References

1. **Westfeld, A. & Pfitzmann, A.** (1999). Attacks on Steganographic Systems. Information Hiding.

2. **Fridrich, J. et al.** (2001). Reliable Detection of LSB Steganography. ACM Workshop on Multimedia and Security.

3. **Dumitrescu, S. et al.** (2003). Detection of LSB Steganography via Sample Pair Analysis. IEEE Transactions on Signal Processing.

4. **Fridrich, J. & Goljan, M.** (2002). Practical Steganalysis of Digital Images. SPIE Electronic Imaging.

5. **Ker, A.D.** (2005). Steganalysis of LSB matching in grayscale images. IEEE Signal Processing Letters.

---

## Complete Example

```python
#!/usr/bin/env python3
"""Complete example of steganographic analysis."""

from pathlib import Path
from imagetrust.forensics.steganography import SteganographyDetector

def analyze_image(image_path: str):
    detector = SteganographyDetector()
    report = detector.analyze(Path(image_path))

    print(f"=== Steganographic Analysis: {image_path} ===\n")
    print(f"Overall probability: {report.overall_probability:.1%}")
    print(f"Confidence level: {report.confidence.value}")
    print(f"Verdict: {'SUSPECT' if report.overall_probability > 0.5 else 'PROBABLY CLEAN'}")

    print("\n--- Results per method ---")
    for method, result in report.method_results.items():
        status = "⚠️" if result.detection_probability > 0.5 else "✓"
        print(f"  {status} {method}: {result.detection_probability:.3f}")
        if result.anomalies:
            for anomaly in result.anomalies[:3]:
                print(f"      - {anomaly}")

    if report.detected_methods:
        print(f"\n⚠️  Methods that detected anomalies: {[m.value for m in report.detected_methods]}")

    return report

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_image(sys.argv[1])
    else:
        print("Usage: python example.py <image_path>")
```

---

*Document generated for ImageTrust - Forensic Detection System.*
