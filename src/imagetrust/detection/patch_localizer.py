"""
Patch-Level AI Localization Module.

Divides an image into overlapping patches and runs the ML ensemble
on each patch to produce a spatial probability heatmap showing
which regions are AI-generated vs authentic.

This enables detection of partial edits (e.g. AI-added objects in
a real photograph) where global classifiers may fail because the
majority of pixels are authentic.
"""

import time
import base64
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image, ImageFilter

from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)

# Module-level singleton for the detector to avoid reloading models
_detector_instance = None


def _get_detector():
    """Get or create the singleton MultiModelDetector."""
    global _detector_instance
    if _detector_instance is None:
        from imagetrust.detection.multi_detector import MultiModelDetector
        _detector_instance = MultiModelDetector()
    return _detector_instance


def localize_ai_regions(
    image: Image.Image,
    patch_size: int = 128,
    stride: int = 64,
    min_image_size: int = 256,
    batch_size: int = 16,
    models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Perform patch-level AI localization on an image.

    Args:
        image: PIL Image to analyze.
        patch_size: Size of each square patch (pixels).
        stride: Step between patches (smaller = more overlap = smoother).
        min_image_size: Minimum dimension to perform localization.
        batch_size: Number of patches to process in one forward pass.
        models: Optional list of specific model IDs to use.

    Returns:
        Dictionary with:
        - heatmap: 2D numpy array of AI probabilities per patch
        - heatmap_base64: Base64-encoded PNG of the colored heatmap
        - overlay_base64: Base64-encoded PNG of heatmap overlaid on image
        - grid_shape: (rows, cols) of the patch grid
        - patch_size: actual patch size used
        - stride: actual stride used
        - hot_regions: list of regions with high AI probability
        - mean_ai_prob: average AI probability across all patches
        - max_ai_prob: maximum AI probability in any patch
        - processing_time_ms: time taken
    """
    import torch

    start_time = time.time()
    w, h = image.size

    # Skip if image is too small
    if min(w, h) < min_image_size:
        return {
            "heatmap": None,
            "heatmap_base64": "",
            "overlay_base64": "",
            "grid_shape": (0, 0),
            "patch_size": patch_size,
            "stride": stride,
            "hot_regions": [],
            "mean_ai_prob": 0.0,
            "max_ai_prob": 0.0,
            "processing_time_ms": 0.0,
            "skipped": True,
            "reason": f"Image too small ({w}x{h})",
        }

    # Adaptive patch/stride for very large images
    if max(w, h) > 2048:
        patch_size = max(patch_size, 192)
        stride = max(stride, 128)
    elif max(w, h) > 4096:
        patch_size = max(patch_size, 256)
        stride = max(stride, 192)

    # Ensure RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Extract patches
    patches = []
    positions = []  # (row_idx, col_idx, x, y)

    row_idx = 0
    for y in range(0, h - patch_size + 1, stride):
        col_idx = 0
        for x in range(0, w - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            positions.append((row_idx, col_idx, x, y))
            col_idx += 1
        row_idx += 1

    if not patches:
        return {
            "heatmap": None,
            "heatmap_base64": "",
            "overlay_base64": "",
            "grid_shape": (0, 0),
            "patch_size": patch_size,
            "stride": stride,
            "hot_regions": [],
            "mean_ai_prob": 0.0,
            "max_ai_prob": 0.0,
            "processing_time_ms": 0.0,
            "skipped": True,
            "reason": "No patches extracted",
        }

    n_rows = max(p[0] for p in positions) + 1
    n_cols = max(p[1] for p in positions) + 1

    logger.info(
        f"Patch localization: {len(patches)} patches "
        f"({n_rows}x{n_cols} grid, patch={patch_size}, stride={stride})"
    )

    # Load ML models (reuse singleton)
    detector = _get_detector()

    if not detector.models:
        return {
            "heatmap": None,
            "heatmap_base64": "",
            "overlay_base64": "",
            "grid_shape": (n_rows, n_cols),
            "patch_size": patch_size,
            "stride": stride,
            "hot_regions": [],
            "mean_ai_prob": 0.0,
            "max_ai_prob": 0.0,
            "processing_time_ms": 0.0,
            "skipped": True,
            "reason": "No ML models loaded",
        }

    # Process patches through all models
    patch_probs = np.zeros(len(patches), dtype=np.float32)

    # Use the fastest 2 models for localization (speed optimization)
    model_items = list(detector.models.items())
    if len(model_items) > 2:
        # Pick first 2 models (typically fastest)
        model_items = model_items[:2]

    for model_id, model_info in model_items:
        processor = model_info["processor"]
        model = model_info["model"]

        # Process in batches
        for batch_start in range(0, len(patches), batch_size):
            batch_patches = patches[batch_start:batch_start + batch_size]

            try:
                inputs = processor(
                    images=batch_patches, return_tensors="pt"
                )
                inputs = {
                    k: v.to(detector.device) for k, v in inputs.items()
                }

                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)

                for i, prob in enumerate(probs):
                    ai_prob = detector._compute_ai_probability(
                        prob, model_info["id2label"]
                    )
                    idx = batch_start + i
                    patch_probs[idx] += ai_prob

            except Exception as e:
                logger.warning(
                    f"Batch {batch_start} failed for {model_id}: {e}"
                )

    # Average across models
    n_models = len(model_items)
    if n_models > 0:
        patch_probs /= n_models

    # --- Outlier-based normalization ---
    # Compression and CG textures cause uniformly elevated probabilities.
    # We only want to flag regions that are *anomalously* high relative to
    # the rest of the image.  Subtract the median and rescale so that only
    # true outliers survive.
    raw_mean = float(np.mean(patch_probs))
    raw_max = float(np.max(patch_probs))
    median_prob = float(np.median(patch_probs))
    std_prob = float(np.std(patch_probs))

    if std_prob > 0.01:
        # Z-score relative to median: patches must be >1.5 std above median
        # to register as meaningfully different from the background.
        normalized = (patch_probs - median_prob) / std_prob
        # Map back to [0, 1]: z=0 -> 0.3, z=2 -> 0.7, z=3+ -> 0.9+
        display_probs = np.clip(0.3 + normalized * 0.2, 0.0, 1.0)
        # For hot-region detection keep the z-scores
        hot_probs = normalized
    else:
        # Uniform image — no variance, nothing to highlight
        display_probs = np.full_like(patch_probs, 0.3)
        hot_probs = np.zeros_like(patch_probs)

    # Build 2D heatmap grid (display uses normalized values)
    heatmap = np.zeros((n_rows, n_cols), dtype=np.float32)
    heatmap_raw = np.zeros((n_rows, n_cols), dtype=np.float32)
    for i, (r, c, x, y) in enumerate(positions):
        heatmap[r, c] = display_probs[i]
        heatmap_raw[r, c] = patch_probs[i]

    # Generate colored heatmap image
    heatmap_img = _probability_to_heatmap(heatmap, (w, h), patch_size, stride)
    overlay_img = _create_overlay(image, heatmap_img, alpha=0.45)

    # Encode to base64
    heatmap_b64 = _image_to_base64(heatmap_img)
    overlay_b64 = _image_to_base64(overlay_img)

    # Find hot regions: must be >5.0 std above median AND raw prob > 0.70
    # Very strict thresholds — only flag patches that are genuine statistical
    # outliers, eliminating false positives from JPEG compression and CG textures.
    hot_regions = _find_hot_regions(
        heatmap_raw, positions, patch_size, threshold=0.70,
        z_scores=hot_probs, z_threshold=5.0,
    )

    elapsed_ms = (time.time() - start_time) * 1000

    return {
        "heatmap": heatmap.tolist(),
        "heatmap_base64": heatmap_b64,
        "overlay_base64": overlay_b64,
        "grid_shape": (n_rows, n_cols),
        "patch_size": patch_size,
        "stride": stride,
        "hot_regions": hot_regions,
        "mean_ai_prob": raw_mean,
        "max_ai_prob": raw_max,
        "processing_time_ms": elapsed_ms,
        "skipped": False,
        "n_patches": len(patches),
        "n_models_used": n_models,
    }


def _probability_to_heatmap(
    heatmap: np.ndarray,
    image_size: Tuple[int, int],
    patch_size: int,
    stride: int,
) -> Image.Image:
    """Convert probability grid to a colored heatmap image."""
    w, h = image_size
    n_rows, n_cols = heatmap.shape

    # Create heatmap at patch grid resolution
    grid_img = np.zeros((n_rows, n_cols, 3), dtype=np.uint8)

    for r in range(n_rows):
        for c in range(n_cols):
            prob = heatmap[r, c]
            grid_img[r, c] = _prob_to_rgb(prob)

    # Upscale to image size with interpolation
    grid_pil = Image.fromarray(grid_img, mode="RGB")
    heatmap_full = grid_pil.resize((w, h), Image.BILINEAR)

    # Smooth for visual appeal
    heatmap_full = heatmap_full.filter(ImageFilter.GaussianBlur(radius=8))

    return heatmap_full


def _prob_to_rgb(prob: float) -> Tuple[int, int, int]:
    """Map AI probability [0,1] to a color gradient.

    0.0 = green (authentic), 0.5 = yellow (uncertain), 1.0 = red (AI).
    """
    prob = max(0.0, min(1.0, prob))

    if prob < 0.5:
        # Green to yellow
        t = prob * 2
        r = int(255 * t)
        g = 200
        b = 0
    else:
        # Yellow to red
        t = (prob - 0.5) * 2
        r = 255
        g = int(200 * (1 - t))
        b = 0

    return (r, g, b)


def _create_overlay(
    original: Image.Image,
    heatmap: Image.Image,
    alpha: float = 0.45,
) -> Image.Image:
    """Blend heatmap over the original image."""
    orig_rgba = original.convert("RGBA")
    heat_rgba = heatmap.convert("RGBA")

    # Adjust heatmap opacity
    heat_data = np.array(heat_rgba, dtype=np.float32)
    heat_data[:, :, 3] = int(255 * alpha)
    heat_rgba = Image.fromarray(heat_data.astype(np.uint8), mode="RGBA")

    # Composite
    composite = Image.alpha_composite(orig_rgba, heat_rgba)
    return composite.convert("RGB")


def _image_to_base64(img: Image.Image) -> str:
    """Encode PIL Image to base64 PNG string."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _find_hot_regions(
    heatmap: np.ndarray,
    positions: List[Tuple[int, int, int, int]],
    patch_size: int,
    threshold: float = 0.55,
    z_scores: Optional[np.ndarray] = None,
    z_threshold: float = 2.0,
) -> List[Dict[str, Any]]:
    """Find regions with anomalously high AI probability.

    A patch must satisfy BOTH conditions to be flagged:
    1. Raw AI probability > threshold (absolute)
    2. Z-score > z_threshold (relative to image median)

    This eliminates uniform false positives from compression/CG textures
    while still catching genuine AI-edited regions that stand out.
    """
    hot = []
    for i, (r, c, x, y) in enumerate(positions):
        prob = heatmap[r, c]
        z = z_scores[i] if z_scores is not None else 999.0

        if prob >= threshold and z >= z_threshold:
            hot.append({
                "row": r,
                "col": c,
                "x": x,
                "y": y,
                "width": patch_size,
                "height": patch_size,
                "ai_probability": float(prob),
                "z_score": float(z),
                "severity": (
                    "critical" if prob > 0.75 and z > 3.0
                    else "warning" if prob > 0.55
                    else "info"
                ),
            })

    # Sort by z-score descending (most anomalous first)
    hot.sort(key=lambda r: r["z_score"], reverse=True)
    return hot[:20]  # Top 20 regions
