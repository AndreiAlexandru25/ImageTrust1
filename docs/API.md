# ImageTrust API Documentation

Complete reference for the ImageTrust REST API and Python SDK.

## Table of Contents

1. [REST API](#rest-api)
2. [Python SDK](#python-sdk)
3. [Response Schemas](#response-schemas)
4. [Error Handling](#error-handling)
5. [Examples](#examples)

---

## REST API

### Base URL

```
Development: http://localhost:8000
Production:  https://api.imagetrust.example.com
```

### Authentication

Currently, the API does not require authentication. For production, implement API keys:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" ...
```

---

### Endpoints

#### `POST /analyze`

Analyze a single image for AI-generated content.

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@image.jpg"
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `file` | File | Yes | Image file (JPG, PNG, WebP) |
| `use_calibration` | bool | No | Apply probability calibration (default: true) |
| `detailed` | bool | No | Include detailed analysis (default: false) |

**Response:**
```json
{
  "ai_probability": 0.87,
  "real_probability": 0.13,
  "verdict": "ai_generated",
  "confidence": "high",
  "calibrated": true,
  "processing_time_ms": 523.4,
  "model_name": "ensemble"
}
```

**Status Codes:**
| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Invalid file format |
| 413 | File too large (>50MB) |
| 500 | Internal server error |

---

#### `POST /analyze/batch`

Analyze multiple images in a single request.

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.png" \
  -F "files=@image3.webp"
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `files` | File[] | Yes | Multiple image files (max 10) |

**Response:**
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "ai_probability": 0.92,
      "verdict": "ai_generated",
      "confidence": "very_high"
    },
    {
      "filename": "image2.png",
      "ai_probability": 0.15,
      "verdict": "real",
      "confidence": "high"
    }
  ],
  "total_processing_time_ms": 1245.7,
  "count": 2
}
```

---

#### `POST /analyze/url`

Analyze an image from URL.

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.jpg"}'
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `url` | string | Yes | Image URL |
| `timeout` | int | No | Download timeout in seconds (default: 30) |

**Response:** Same as `/analyze`

---

#### `POST /analyze/detailed`

Get comprehensive forensic analysis.

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze/detailed" \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "verdict": "ai_generated",
  "ai_probability": 0.87,
  "real_probability": 0.13,
  "confidence": "high",
  "calibrated": true,

  "model_results": [
    {
      "model": "umm-maybe/AI-image-detector",
      "ai_probability": 0.91,
      "weight": 0.25
    },
    {
      "model": "Organika/sdxl-detector",
      "ai_probability": 0.85,
      "weight": 0.25
    }
  ],

  "signal_analysis": {
    "frequency": {
      "score": 0.72,
      "description": "High-frequency artifacts detected"
    },
    "noise": {
      "score": 0.65,
      "description": "Uniform noise pattern (typical of AI)"
    },
    "texture": {
      "score": 0.58,
      "description": "Smooth texture regions detected"
    }
  },

  "generator_identification": {
    "primary_generator": "stable_diffusion_xl",
    "confidence": 0.73,
    "all_scores": {
      "stable_diffusion_xl": 0.73,
      "midjourney_v6": 0.15,
      "dalle_3": 0.08
    }
  },

  "metadata": {
    "has_exif": false,
    "software": null,
    "camera_model": null,
    "creation_date": null
  },

  "image_info": {
    "width": 1024,
    "height": 1024,
    "format": "JPEG",
    "file_size_bytes": 245678,
    "sha256": "abc123..."
  },

  "processing_time_ms": 1523.4
}
```

---

#### `GET /health`

Health check endpoint.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": true,
  "gpu_available": true
}
```

---

#### `GET /info`

System information.

**Request:**
```bash
curl http://localhost:8000/info
```

**Response:**
```json
{
  "version": "1.0.0",
  "environment": "production",
  "python_version": "3.10.12",
  "torch_version": "2.1.0",
  "cuda_available": true,
  "cuda_version": "11.8",
  "models": [
    "umm-maybe/AI-image-detector",
    "Organika/sdxl-detector",
    "aiornot/aiornot-detector-v2",
    "nyuad/ai-image-detector-2025"
  ]
}
```

---

## Python SDK

### Installation

```bash
pip install imagetrust
```

### Quick Start

```python
from imagetrust.detection import AIDetector

# Initialize detector
detector = AIDetector()

# Analyze single image
result = detector.detect("image.jpg")
print(f"AI Probability: {result['ai_probability']:.1%}")
print(f"Verdict: {result['verdict'].value}")
```

### AIDetector Class

```python
class AIDetector:
    def __init__(
        self,
        model: str = "ensemble",
        device: str = "auto",
        use_calibration: bool = True,
    ):
        """
        Initialize AI image detector.

        Args:
            model: Model to use ("ensemble", "efficientnet", "vit", etc.)
            device: Computation device ("auto", "cuda", "cpu")
            use_calibration: Whether to apply probability calibration
        """
```

**Methods:**

```python
def detect(
    self,
    image: Union[str, Path, bytes, Image.Image],
    use_calibration: bool = True,
) -> Dict[str, Any]:
    """
    Detect if an image is AI-generated.

    Args:
        image: Image path, bytes, or PIL Image
        use_calibration: Apply calibration to probabilities

    Returns:
        Dictionary with detection results
    """

def detect_batch(
    self,
    images: List[Union[str, Path]],
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """
    Batch detection for multiple images.

    Args:
        images: List of image paths
        batch_size: Processing batch size

    Returns:
        List of detection results
    """
```

### ComprehensiveDetector Class

```python
from imagetrust.detection.multi_detector import ComprehensiveDetector

detector = ComprehensiveDetector()

# Full analysis with all components
result = detector.analyze(image)

# Access individual components
print(result.verdict)           # DetectionVerdict enum
print(result.ai_probability)    # 0.0 - 1.0
print(result.confidence)        # Confidence enum
print(result.model_results)     # Individual model outputs
print(result.signal_analysis)   # Frequency/noise/texture analysis
```

### Baseline Detectors

```python
from imagetrust.baselines import get_baseline, list_baselines

# List available baselines
print(list_baselines())  # ['classical', 'cnn', 'vit', 'imagetrust']

# Get specific baseline
classical = get_baseline("classical")
cnn = get_baseline("cnn", backbone="resnet50")
vit = get_baseline("vit", architecture="clip")

# Train baseline
cnn.fit(train_images, train_labels, val_images, val_labels)

# Predict
result = cnn.predict_proba(image)
print(result.ai_probability)
```

### Calibration

```python
from imagetrust.baselines import BaselineCalibrator, calibrate_baseline

# Calibrate a baseline
calibrator, cal_result = calibrate_baseline(
    baseline,
    val_images,
    val_labels,
    method="temperature"
)

print(f"ECE before: {cal_result.ece_before:.4f}")
print(f"ECE after: {cal_result.ece_after:.4f}")

# Apply calibration
calibrated_prob = calibrator.calibrate(raw_probability)
```

### Uncertainty Estimation

```python
from imagetrust.baselines import SelectivePredictor

# Wrap baseline with selective prediction
predictor = SelectivePredictor(
    baseline,
    uncertainty_method="entropy",
    target_coverage=0.9
)

# Calibrate threshold on validation data
predictor.calibrate(val_images, val_labels)

# Predict with uncertainty
result = predictor.predict(image)
if result.should_abstain:
    print("UNCERTAIN - model abstains from prediction")
else:
    print(f"Prediction: {'AI' if result.ai_probability > 0.5 else 'Real'}")
```

---

## Response Schemas

### DetectionResult

```python
@dataclass
class DetectionResult:
    ai_probability: float        # 0.0 - 1.0
    real_probability: float      # 1 - ai_probability
    verdict: DetectionVerdict    # REAL, AI_GENERATED, UNCERTAIN
    confidence: Confidence       # VERY_LOW to VERY_HIGH
    calibrated: bool            # Whether calibration was applied
    model_name: str             # Model used
    processing_time_ms: float   # Processing time
```

### DetectionVerdict (Enum)

```python
class DetectionVerdict(Enum):
    REAL = "real"
    AI_GENERATED = "ai_generated"
    MANIPULATED = "manipulated"
    UNCERTAIN = "uncertain"
```

### Confidence (Enum)

```python
class Confidence(Enum):
    VERY_LOW = "very_low"    # < 55%
    LOW = "low"              # 55-65%
    MEDIUM = "medium"        # 65-75%
    HIGH = "high"            # 75-90%
    VERY_HIGH = "very_high"  # > 90%
```

### BaselineResult

```python
@dataclass
class BaselineResult:
    ai_probability: float
    real_probability: float
    raw_logits: Optional[np.ndarray]
    raw_probability: Optional[float]
    baseline_name: str
    processing_time_ms: float
    calibrated: bool
    features: Optional[np.ndarray]
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_FILE_FORMAT",
    "message": "Unsupported file format. Supported: JPG, PNG, WebP",
    "details": {
      "received_format": "GIF",
      "supported_formats": ["jpg", "jpeg", "png", "webp"]
    }
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_FILE_FORMAT` | 400 | Unsupported image format |
| `FILE_TOO_LARGE` | 413 | File exceeds size limit |
| `INVALID_URL` | 400 | Could not fetch image from URL |
| `MODEL_ERROR` | 500 | Model inference failed |
| `TIMEOUT` | 504 | Request timed out |

### Python Exception Handling

```python
from imagetrust.exceptions import (
    ImageTrustError,
    InvalidImageError,
    ModelNotFoundError,
    CalibrationError,
)

try:
    result = detector.detect(image)
except InvalidImageError as e:
    print(f"Invalid image: {e}")
except ModelNotFoundError as e:
    print(f"Model not found: {e}")
except ImageTrustError as e:
    print(f"ImageTrust error: {e}")
```

---

## Examples

### cURL Examples

```bash
# Basic analysis
curl -X POST http://localhost:8000/analyze \
  -F "file=@photo.jpg"

# Detailed analysis
curl -X POST http://localhost:8000/analyze/detailed \
  -F "file=@photo.jpg" | jq .

# Batch analysis
curl -X POST http://localhost:8000/analyze/batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -F "files=@img3.jpg"

# URL analysis
curl -X POST http://localhost:8000/analyze/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.jpg"}'
```

### Python Examples

```python
# Basic usage
from imagetrust.detection import AIDetector

detector = AIDetector()
result = detector.detect("suspicious_image.jpg")

if result["verdict"].value == "ai_generated":
    print(f"AI detected with {result['ai_probability']:.1%} confidence")

# Batch processing
from pathlib import Path

images = list(Path("images/").glob("*.jpg"))
results = detector.detect_batch(images, batch_size=8)

ai_count = sum(1 for r in results if r["verdict"].value == "ai_generated")
print(f"Found {ai_count}/{len(images)} AI-generated images")

# With calibration comparison
from imagetrust.baselines import get_baseline, BaselineCalibrator

baseline = get_baseline("cnn")
baseline.fit(train_images, train_labels, val_images, val_labels)

# Compare calibration methods
calibrator = BaselineCalibrator(method="temperature")
calibrator.fit(val_probs, val_labels)

raw_result = baseline.predict_proba(test_image)
calibrated = calibrator.calibrate(raw_result.ai_probability)

print(f"Raw: {raw_result.ai_probability:.3f}")
print(f"Calibrated: {calibrated:.3f}")
```

### Integration Examples

**Flask Integration:**
```python
from flask import Flask, request, jsonify
from imagetrust.detection import AIDetector

app = Flask(__name__)
detector = AIDetector()

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    image_bytes = file.read()
    result = detector.detect(image_bytes)
    return jsonify({
        'ai_probability': result['ai_probability'],
        'verdict': result['verdict'].value,
    })
```

**Async Processing:**
```python
import asyncio
from imagetrust.detection import AIDetector

async def analyze_images(paths):
    detector = AIDetector()

    async def analyze_one(path):
        return detector.detect(path)

    tasks = [analyze_one(p) for p in paths]
    return await asyncio.gather(*tasks)

# Run
results = asyncio.run(analyze_images(image_paths))
```

---

## Rate Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/analyze` | 60 requests | per minute |
| `/analyze/batch` | 10 requests | per minute |
| `/analyze/url` | 30 requests | per minute |

---

## OpenAPI Specification

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`
