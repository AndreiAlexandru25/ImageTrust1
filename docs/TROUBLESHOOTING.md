# ImageTrust Troubleshooting Guide

Solutions for common issues when installing, running, or deploying ImageTrust.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Model Loading Problems](#model-loading-problems)
3. [GPU/CUDA Issues](#gpucuda-issues)
4. [Desktop Application Issues](#desktop-application-issues)
5. [API Server Issues](#api-server-issues)
6. [Baseline Training Issues](#baseline-training-issues)
7. [Calibration Issues](#calibration-issues)
8. [Performance Issues](#performance-issues)
9. [Build/Packaging Issues](#buildpackaging-issues)

---

## Installation Issues

### `pip install` fails with dependency conflicts

**Symptom:**
```
ERROR: Cannot install imagetrust because these package versions have conflicting dependencies.
```

**Solution:**
```bash
# Create fresh virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install with clean slate
pip install --upgrade pip
pip install -e .
```

### `ModuleNotFoundError: No module named 'imagetrust'`

**Symptom:** Import fails after installation.

**Solutions:**

1. Ensure you installed in development mode:
   ```bash
   pip install -e .
   ```

2. Check Python path:
   ```python
   import sys
   print(sys.path)
   # Ensure project src/ directory is included
   ```

3. Reinstall:
   ```bash
   pip uninstall imagetrust
   pip install -e .
   ```

### Wrong Python version

**Symptom:**
```
SyntaxError: invalid syntax
# or
TypeError: 'type' object is not subscriptable
```

**Solution:**
ImageTrust requires Python 3.10+.

```bash
python --version  # Check version
# If < 3.10, install newer Python

# Use specific Python version
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
```

---

## Model Loading Problems

### Models fail to download

**Symptom:**
```
OSError: We couldn't connect to 'https://huggingface.co' to load this model
```

**Solutions:**

1. Check internet connection
2. Try manual download:
   ```bash
   pip install huggingface_hub
   huggingface-cli download umm-maybe/AI-image-detector
   ```

3. Use offline mode if models already cached:
   ```bash
   export TRANSFORMERS_OFFLINE=1
   ```

4. Behind corporate proxy:
   ```bash
   export HTTP_PROXY=http://proxy:port
   export HTTPS_PROXY=http://proxy:port
   ```

### `RuntimeError: CUDA out of memory`

**Symptom:** Out of memory when loading multiple models.

**Solutions:**

1. Load models sequentially (not all at once):
   ```python
   # Process one model at a time
   for model in models:
       result = model.predict(image)
       del model  # Free memory
       torch.cuda.empty_cache()
   ```

2. Use CPU for some models:
   ```python
   detector = AIDetector(device="cpu")
   ```

3. Reduce batch size:
   ```python
   results = detector.detect_batch(images, batch_size=1)
   ```

### `KeyError` or `ValueError` when loading model

**Symptom:**
```
KeyError: 'model.classifier.weight'
# or
ValueError: size mismatch for classifier.weight
```

**Solution:**
Model cache may be corrupted. Clear and re-download:

```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/hub/models--umm-maybe--AI-image-detector

# Re-run to trigger fresh download
python -c "from imagetrust.detection import AIDetector; AIDetector()"
```

---

## GPU/CUDA Issues

### `torch.cuda.is_available()` returns `False`

**Symptom:** GPU not detected even though NVIDIA GPU exists.

**Diagnostic:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
```

**Solutions:**

1. Install CUDA-enabled PyTorch:
   ```bash
   # Check your CUDA version
   nvidia-smi

   # Install matching PyTorch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   # or cu121 for CUDA 12.1
   ```

2. NVIDIA driver outdated:
   ```bash
   # Check driver version
   nvidia-smi
   # Update via NVIDIA website or package manager
   ```

3. WSL2 users:
   ```bash
   # Ensure CUDA is properly configured for WSL
   # Install NVIDIA CUDA Toolkit for WSL
   ```

### CUDA version mismatch

**Symptom:**
```
RuntimeError: CUDA error: no kernel image is available for execution
```

**Solution:**
PyTorch CUDA version must match installed CUDA:

```bash
# Check installed CUDA
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"

# If mismatch, reinstall PyTorch with correct CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### cuDNN errors

**Symptom:**
```
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED
```

**Solutions:**

1. Disable cuDNN (slower but works):
   ```python
   torch.backends.cudnn.enabled = False
   ```

2. Set deterministic mode:
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

---

## Desktop Application Issues

### Application won't start

**Symptom:** Double-clicking .exe does nothing or crashes immediately.

**Solutions:**

1. Run from command line to see errors:
   ```cmd
   cd dist\ImageTrust
   ImageTrust.exe
   ```

2. Missing Visual C++ Redistributable:
   - Download and install from Microsoft

3. Check Windows Event Viewer for crash logs

### Qt platform plugin error

**Symptom:**
```
qt.qpa.plugin: Could not find the Qt platform plugin "windows"
```

**Solutions:**

1. Set platform path:
   ```cmd
   set QT_QPA_PLATFORM_PLUGIN_PATH=dist\ImageTrust\_internal\PySide6\plugins\platforms
   ```

2. Rebuild with explicit plugin inclusion:
   ```python
   # In ImageTrust.spec
   datas += collect_data_files('PySide6', include_py_files=False)
   ```

### Drag and drop not working

**Symptom:** Dropping images on the application does nothing.

**Solutions:**

1. Run as administrator (Windows UAC can block drag-drop)
2. Check file type (only JPG, PNG, WebP supported)
3. Verify file is not too large (>50MB)

### Application freezes during analysis

**Symptom:** UI becomes unresponsive.

**Cause:** Analysis running on main thread.

**Solution:** Analysis should run in background thread. If using modified code, ensure QThread is used:

```python
class AnalysisWorker(QThread):
    finished = Signal(dict)

    def run(self):
        result = detector.detect(self.image_path)
        self.finished.emit(result)
```

---

## API Server Issues

### Port already in use

**Symptom:**
```
OSError: [Errno 98] Address already in use
```

**Solutions:**

1. Use different port:
   ```bash
   imagetrust serve --port 8001
   ```

2. Kill existing process:
   ```bash
   # Linux
   lsof -i :8000 | grep LISTEN
   kill -9 <PID>

   # Windows
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   ```

### Request timeout

**Symptom:**
```
504 Gateway Timeout
```

**Solutions:**

1. Increase server timeout:
   ```bash
   uvicorn imagetrust.api.main:app --timeout-keep-alive 120
   ```

2. For Nginx proxy:
   ```nginx
   proxy_read_timeout 120s;
   proxy_connect_timeout 120s;
   ```

3. For large images, consider async processing:
   ```python
   @app.post("/analyze/async")
   async def analyze_async(file: UploadFile):
       task_id = create_background_task(file)
       return {"task_id": task_id, "status": "processing"}
   ```

### CORS errors

**Symptom:**
```
Access to fetch has been blocked by CORS policy
```

**Solution:**
Add CORS middleware:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend
    allow_methods=["POST"],
    allow_headers=["*"],
)
```

### File upload size limit

**Symptom:**
```
413 Request Entity Too Large
```

**Solutions:**

1. Nginx configuration:
   ```nginx
   client_max_body_size 50M;
   ```

2. FastAPI limit:
   ```python
   @app.post("/analyze")
   async def analyze(file: UploadFile = File(..., max_length=50*1024*1024)):
       ...
   ```

---

## Baseline Training Issues

### Training runs out of memory

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**

1. Reduce batch size:
   ```python
   baseline.fit(train_images, train_labels, batch_size=8)  # or 4, or 2
   ```

2. Enable gradient checkpointing:
   ```python
   baseline = get_baseline("cnn", gradient_checkpointing=True)
   ```

3. Use mixed precision:
   ```python
   from torch.cuda.amp import autocast
   with autocast():
       loss = model(batch)
   ```

4. Clear cache periodically:
   ```python
   torch.cuda.empty_cache()
   ```

### Training loss not decreasing

**Symptom:** Loss stays flat or increases.

**Solutions:**

1. Check learning rate:
   ```python
   # Try smaller learning rate
   baseline.fit(..., learning_rate=1e-5)
   ```

2. Check data loading:
   ```python
   # Verify labels are correct
   for img, label in train_loader:
       print(f"Label distribution: {label.sum()}/{len(label)}")
       break
   ```

3. Check for data leakage:
   ```python
   # Ensure no overlap between train/val/test
   train_set = set(train_paths)
   val_set = set(val_paths)
   assert len(train_set & val_set) == 0, "Data leakage detected!"
   ```

### NaN loss during training

**Symptom:**
```
Loss: nan
```

**Solutions:**

1. Gradient clipping:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. Check for extreme values in input:
   ```python
   # Normalize images properly
   transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
   ```

3. Lower learning rate:
   ```python
   baseline.fit(..., learning_rate=1e-6)
   ```

---

## Calibration Issues

### ECE not improving

**Symptom:** Expected Calibration Error remains high after calibration.

**Solutions:**

1. Try different calibration method:
   ```python
   # Temperature scaling (simple)
   calibrator = BaselineCalibrator(method="temperature")

   # Isotonic regression (more flexible)
   calibrator = BaselineCalibrator(method="isotonic")
   ```

2. Check validation set size:
   ```python
   # Need sufficient samples for calibration
   assert len(val_probs) >= 100, "Too few samples for calibration"
   ```

3. Check probability distribution:
   ```python
   import matplotlib.pyplot as plt
   plt.hist(val_probs, bins=20)
   plt.title("Uncalibrated probability distribution")
   plt.show()
   # If all near 0 or 1, model is overconfident
   ```

### Calibration makes predictions worse

**Symptom:** Accuracy drops after calibration.

**Cause:** Calibration should NOT change accuracy, only confidence estimates.

**Diagnostic:**
```python
# Calibration should not change predictions
raw_preds = (raw_probs > 0.5).astype(int)
cal_preds = (cal_probs > 0.5).astype(int)
assert np.array_equal(raw_preds, cal_preds), "Calibration changed predictions!"
```

**Solution:**
If accuracy changes, there may be a bug in calibration application. Check threshold handling.

---

## Performance Issues

### Inference too slow

**Symptom:** Single image takes >5 seconds.

**Solutions:**

1. Verify GPU is being used:
   ```python
   import torch
   print(f"Using: {'GPU' if torch.cuda.is_available() else 'CPU'}")
   ```

2. Use batch processing:
   ```python
   # Instead of loop
   results = detector.detect_batch(images, batch_size=8)
   ```

3. Reduce model count:
   ```python
   # Use single model instead of ensemble
   detector = AIDetector(model="efficientnet")
   ```

4. Enable model caching:
   ```python
   # Models loaded once, reused
   detector = AIDetector()  # Keep reference
   for img in images:
       result = detector.detect(img)  # Reuses loaded models
   ```

### High memory usage

**Symptom:** RAM usage grows over time.

**Solutions:**

1. Process in batches and clear:
   ```python
   for batch in batches:
       results = detector.detect_batch(batch)
       process_results(results)
       del results
       gc.collect()
   ```

2. Limit model cache:
   ```python
   import os
   os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
   # Clear periodically
   ```

---

## Build/Packaging Issues

### PyInstaller build fails

**Symptom:**
```
ModuleNotFoundError during PyInstaller build
```

**Solution:**
Add missing module to spec file:

```python
# ImageTrust.spec
hiddenimports = [
    ...,
    'missing_module',
]
```

### Executable too large (>1GB)

**Symptom:** Built .exe is unexpectedly large.

**Solutions:**

1. Check what's included:
   ```bash
   pyinstaller ImageTrust.spec --debug=imports 2>&1 | grep "import"
   ```

2. Add exclusions:
   ```python
   # ImageTrust.spec
   excludes = [
       'matplotlib',
       'notebook',
       'pytest',
       'sphinx',
   ]
   ```

3. Use UPX compression:
   ```bash
   pyinstaller ImageTrust.spec --upx-dir=/path/to/upx
   ```

### DLL not found on Windows

**Symptom:**
```
ImportError: DLL load failed while importing...
```

**Solutions:**

1. Install Visual C++ Redistributable 2015-2022
2. Check DLL dependencies:
   ```bash
   # Use Dependency Walker or dumpbin
   dumpbin /dependents ImageTrust.exe
   ```

3. Include missing DLLs in spec:
   ```python
   # ImageTrust.spec
   binaries = [
       ('path/to/missing.dll', '.'),
   ]
   ```

---

## Getting Help

If your issue isn't covered here:

1. **Check existing issues:**
   https://github.com/imagetrust/imagetrust/issues

2. **Open new issue with:**
   - Python version: `python --version`
   - OS: Windows/Linux/Mac version
   - CUDA version: `nvidia-smi`
   - PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - Full error traceback
   - Minimal reproduction steps

3. **Include diagnostic output:**
   ```python
   from imagetrust.utils.diagnostics import print_system_info
   print_system_info()
   ```

---

## Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| Import error | `pip install -e .` |
| CUDA not found | Check PyTorch CUDA version matches |
| Out of memory | Reduce batch size to 1 |
| Model download fails | Check internet, try proxy |
| Slow inference | Verify GPU is being used |
| Port in use | Use `--port 8001` |
| .exe won't start | Run from cmd to see error |
| NaN loss | Lower learning rate, clip gradients |
| ECE high | Try isotonic calibration |
