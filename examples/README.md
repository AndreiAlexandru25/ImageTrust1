# ImageTrust Examples

This directory contains example scripts demonstrating how to use ImageTrust for various tasks.

## Quick Start

```bash
# Install ImageTrust
pip install -e .

# Run any example
python examples/01_basic_detection.py --demo
```

## Examples

### 1. Basic Detection (`01_basic_detection.py`)

Simple AI image detection using the high-level API.

```bash
# With your own image
python examples/01_basic_detection.py path/to/image.jpg

# With demo image
python examples/01_basic_detection.py --demo
```

**What it demonstrates:**
- Using `AIDetector` for single image analysis
- Using the baseline framework
- Direct PIL Image input

### 2. Batch Processing (`02_batch_processing.py`)

Efficiently process multiple images.

```bash
# Process a directory of images
python examples/02_batch_processing.py ./my_images/

# With demo images
python examples/02_batch_processing.py --demo

# Save results to JSON
python examples/02_batch_processing.py ./my_images/ -o results.json
```

**What it demonstrates:**
- Batch image processing
- Progress tracking
- Result aggregation and export

### 3. Baseline Comparison (`03_baseline_comparison.py`)

Compare different detection baselines.

```bash
# Compare on your labeled data
python examples/03_baseline_comparison.py ./labeled_data/

# With demo data
python examples/03_baseline_comparison.py --demo
```

**Expected data structure:**
```
labeled_data/
├── real/
│   ├── image1.jpg
│   └── image2.jpg
└── ai/
    ├── image3.jpg
    └── image4.jpg
```

**What it demonstrates:**
- Loading multiple baselines
- Computing accuracy and AUC metrics
- Comparison table generation

### 4. Calibration Demo (`04_calibration_demo.py`)

Understand probability calibration.

```bash
python examples/04_calibration_demo.py
```

**What it demonstrates:**
- Expected Calibration Error (ECE)
- Temperature scaling
- Why calibration matters for reliable confidence

## Common Patterns

### Basic Detection

```python
from imagetrust.detection import AIDetector

detector = AIDetector()
result = detector.detect("image.jpg")

print(f"AI Probability: {result['ai_probability']:.1%}")
print(f"Verdict: {result['verdict'].value}")
```

### Using Baselines

```python
from imagetrust.baselines import get_baseline

baseline = get_baseline("imagetrust")
result = baseline.predict_proba("image.jpg")

print(f"AI Probability: {result.ai_probability:.1%}")
```

### Batch Processing

```python
from imagetrust.detection import AIDetector
from pathlib import Path

detector = AIDetector()
images = list(Path("./images").glob("*.jpg"))

for img in images:
    result = detector.detect(str(img))
    print(f"{img.name}: {result['ai_probability']:.1%}")
```

### With Calibration

```python
from imagetrust.baselines import get_baseline, calibrate_baseline

baseline = get_baseline("cnn")
baseline.fit(train_images, train_labels, val_images, val_labels)

# Calibrate on validation data
calibrator, cal_result = calibrate_baseline(
    baseline, val_images, val_labels, method="temperature"
)

print(f"ECE improved: {cal_result.ece_before:.4f} → {cal_result.ece_after:.4f}")
```

## Notes

- All examples support `--demo` flag to run without external data
- Examples clean up temporary files automatically
- For production use, see the API documentation in `docs/API.md`
