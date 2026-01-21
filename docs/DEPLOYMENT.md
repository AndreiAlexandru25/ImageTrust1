# ImageTrust Deployment Guide

This guide covers deploying ImageTrust as a Windows executable (.exe) and other deployment scenarios.

## Table of Contents

1. [Windows Desktop Application (.exe)](#windows-desktop-application)
2. [Docker Deployment](#docker-deployment)
3. [API Server Deployment](#api-server-deployment)
4. [Model Weights Management](#model-weights-management)
5. [Troubleshooting](#troubleshooting)

---

## Windows Desktop Application

### Prerequisites

```bash
# Python 3.10+ required
python --version

# Install ImageTrust with desktop dependencies
pip install -e ".[desktop]"

# Install PyInstaller
pip install pyinstaller>=6.0.0
```

### Quick Build

```bash
# Build folder distribution (recommended)
python scripts/build_desktop.py

# Output: dist/ImageTrust/ImageTrust.exe
```

### Build Options

```bash
# Single-file executable (larger, slower startup)
python scripts/build_desktop.py --onefile

# Keep previous build artifacts
python scripts/build_desktop.py --no-clean
```

### Manual PyInstaller Build

```bash
# Using the spec file directly
pyinstaller ImageTrust.spec

# With additional options
pyinstaller ImageTrust.spec --noconfirm --clean
```

### Build Output Structure

**Folder distribution (default):**
```
dist/ImageTrust/
├── ImageTrust.exe          # Main executable
├── _internal/              # Python runtime and dependencies
│   ├── torch/
│   ├── transformers/
│   ├── PySide6/
│   └── ...
└── configs/                # Configuration files
```

**Single-file distribution:**
```
dist/ImageTrust.exe         # Self-contained executable (~500MB+)
```

### Adding Application Icon

1. Create icon file with multiple sizes (16x16 to 256x256):
   ```bash
   # Using ImageMagick
   convert logo.png -define icon:auto-resize=256,128,64,48,32,16 assets/icon.ico
   ```

2. Place in `assets/icon.ico`

3. Rebuild:
   ```bash
   python scripts/build_desktop.py
   ```

### Distribution Checklist

Before distributing the .exe:

- [ ] Test on a clean Windows machine (no Python installed)
- [ ] Verify all models load correctly
- [ ] Test with various image formats (JPG, PNG, WebP)
- [ ] Check file size is reasonable (~500MB-1GB)
- [ ] Verify no console window appears
- [ ] Test drag-and-drop functionality
- [ ] Confirm export to JSON works

### Code Signing (Optional)

For trusted distribution, sign the executable:

```powershell
# Using signtool (Windows SDK)
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com dist\ImageTrust\ImageTrust.exe
```

---

## Docker Deployment

### Build Docker Image

```bash
cd imagetrust
docker build -t imagetrust:latest .
```

### Run Container

```bash
# API server
docker run -p 8000:8000 imagetrust:latest serve

# Web UI
docker run -p 8501:8501 imagetrust:latest ui

# With GPU support
docker run --gpus all -p 8000:8000 imagetrust:latest serve
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  imagetrust-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - IMAGETRUST_ENVIRONMENT=production
      - IMAGETRUST_LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  imagetrust-ui:
    build: .
    command: ui
    ports:
      - "8501:8501"
    depends_on:
      - imagetrust-api
```

---

## API Server Deployment

### Development

```bash
# Local development with auto-reload
imagetrust serve --reload --port 8000
```

### Production with Gunicorn

```bash
pip install gunicorn

# Multi-worker deployment
gunicorn imagetrust.api.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120
```

### Production with Uvicorn

```bash
uvicorn imagetrust.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --loop uvloop \
    --http httptools
```

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/imagetrust
upstream imagetrust {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name imagetrust.example.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://imagetrust;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
    }
}
```

### Environment Variables

```bash
# Production configuration
export IMAGETRUST_ENVIRONMENT=production
export IMAGETRUST_LOG_LEVEL=WARNING
export IMAGETRUST_MODELS_DIR=/opt/imagetrust/models
export IMAGETRUST_OUTPUTS_DIR=/opt/imagetrust/outputs
export IMAGETRUST_CACHE_ENABLED=true
export IMAGETRUST_MAX_UPLOAD_SIZE=52428800  # 50MB
```

---

## Model Weights Management

### Automatic Download

Models are automatically downloaded from HuggingFace on first use:

```python
from imagetrust.detection import AIDetector
detector = AIDetector()  # Downloads models to ~/.cache/huggingface/
```

### Pre-download Models

```bash
# Download all models before deployment
python -c "
from imagetrust.detection.multi_detector import ComprehensiveDetector
detector = ComprehensiveDetector()
print('All models downloaded successfully')
"
```

### Custom Model Directory

```bash
# Set custom cache directory
export HF_HOME=/opt/models/huggingface
export TRANSFORMERS_CACHE=/opt/models/transformers

# Or in Python
import os
os.environ['HF_HOME'] = '/opt/models/huggingface'
```

### Offline Deployment

1. Download models on a machine with internet:
   ```bash
   python -c "from imagetrust.detection.multi_detector import ComprehensiveDetector; ComprehensiveDetector()"
   ```

2. Copy the cache directory:
   ```bash
   # Default location
   cp -r ~/.cache/huggingface /path/to/offline/machine/
   ```

3. Set environment variable on offline machine:
   ```bash
   export HF_HOME=/path/to/offline/machine/huggingface
   export TRANSFORMERS_OFFLINE=1
   ```

### Model Versions

Current models used (as of v1.0.0):
- `umm-maybe/AI-image-detector`
- `Organika/sdxl-detector`
- `aiornot/aiornot-detector-v2`
- `nyuad/ai-image-detector-2025`

---

## Performance Optimization

### GPU Acceleration

```python
# Force GPU usage
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

from imagetrust.detection import AIDetector
detector = AIDetector(device=device)
```

### Batch Processing

```python
# Process multiple images efficiently
from imagetrust.detection import AIDetector

detector = AIDetector()
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = detector.detect_batch(images, batch_size=8)
```

### Memory Optimization

```bash
# Limit GPU memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use mixed precision
export IMAGETRUST_USE_FP16=true
```

---

## Troubleshooting

### Common Build Issues

**"ModuleNotFoundError" during PyInstaller build:**
```bash
# Add missing module to hiddenimports in ImageTrust.spec
hiddenimports = [..., 'missing_module']
```

**Large executable size (>1GB):**
```bash
# Check what's included
pyinstaller ImageTrust.spec --collect-all torch --debug=imports
```

**"DLL load failed" on Windows:**
- Install Visual C++ Redistributable
- Ensure CUDA toolkit matches PyTorch version

### Runtime Issues

**Models not loading:**
```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface
python -c "from imagetrust.detection import AIDetector; AIDetector()"
```

**Out of memory:**
```bash
# Reduce batch size
export IMAGETRUST_BATCH_SIZE=1

# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""
```

**Slow inference:**
- Enable GPU: `torch.cuda.is_available()` should return `True`
- Use batch processing for multiple images
- Consider quantized models for CPU deployment

---

## Security Considerations

### API Security

```python
# Enable CORS for specific origins only
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_methods=["POST"],
)
```

### Rate Limiting

```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze(file: UploadFile):
    ...
```

### Input Validation

- Maximum file size enforced (default: 50MB)
- Supported formats validated (JPG, PNG, WebP)
- Image dimensions checked before processing

---

## Monitoring

### Health Check Endpoint

```bash
curl http://localhost:8000/health
# {"status": "healthy", "version": "1.0.0"}
```

### Prometheus Metrics

```bash
curl http://localhost:8000/metrics
# imagetrust_requests_total{method="POST",endpoint="/analyze"} 1234
# imagetrust_inference_seconds{model="ensemble"} 0.523
```

### Logging

```bash
# Enable debug logging
export IMAGETRUST_LOG_LEVEL=DEBUG

# Log to file
export IMAGETRUST_LOG_FILE=/var/log/imagetrust/app.log
```
