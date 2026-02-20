"""
AI Detection Pack.

Integrates AI-generated image detection:
- HuggingFace pretrained models
- Frequency-domain AI fingerprints
- Falls back gracefully if models unavailable
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from imagetrust.forensics.base import (
    Artifact,
    Confidence,
    ForensicsPlugin,
    ForensicsResult,
    PluginCategory,
    register_plugin,
)
from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


@register_plugin
class HuggingFaceAIDetector(ForensicsPlugin):
    """
    AI detection using HuggingFace pretrained models.

    Uses ensemble of top-rated models for AI image detection.
    Falls back gracefully if models unavailable (no network, no GPU).
    """

    plugin_id = "hf_ai_detector"
    plugin_name = "AI Detection (HuggingFace)"
    category = PluginCategory.AI_DETECTION
    description = "Detects AI-generated images using pretrained models"
    version = "1.0.0"

    # Available models - ordered by reliability
    MODELS = [
        ("umm-maybe/AI-image-detector", "AI Image Detector"),
        ("Nahrawy/AIorNot", "AIorNot"),
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._models_loaded = False
        self._models = {}
        self._device = "cpu"

    def _ensure_models_loaded(self):
        """Lazy load models on first use."""
        if self._models_loaded:
            return

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            for model_id, name in self.MODELS[:2]:  # Load top 2 models
                try:
                    logger.info(f"Loading {name}...")
                    processor = AutoImageProcessor.from_pretrained(model_id)
                    model = AutoModelForImageClassification.from_pretrained(model_id)
                    model.to(self._device)
                    model.eval()

                    self._models[model_id] = {
                        "processor": processor,
                        "model": model,
                        "name": name,
                        "id2label": model.config.id2label,
                    }
                    logger.info(f"Loaded: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")

            self._models_loaded = True

        except ImportError as e:
            logger.warning(f"HuggingFace not available: {e}")
            self._models_loaded = True

    def analyze(
        self,
        image: Image.Image,
        image_path: Optional[Path] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> ForensicsResult:
        """Detect AI-generated images."""
        start_time = time.perf_counter()

        try:
            self._ensure_models_loaded()

            if not self._models:
                return self._create_result(
                    score=0.0,
                    confidence=Confidence.VERY_LOW,
                    detected=False,
                    explanation="AI detection models not available (install transformers or check network)",
                    limitations=["Models could not be loaded"],
                    details={"models_available": False},
                )

            import torch

            # Ensure RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Run each model
            model_results = []

            for model_id, model_info in self._models.items():
                try:
                    processor = model_info["processor"]
                    model = model_info["model"]

                    inputs = processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1)[0]

                    # Find AI class index
                    ai_idx = self._find_ai_index(model_info["id2label"])
                    ai_prob = probs[ai_idx].item()

                    model_results.append({
                        "name": model_info["name"],
                        "ai_probability": ai_prob,
                    })

                except Exception as e:
                    logger.warning(f"Model {model_id} failed: {e}")

            if not model_results:
                return self._create_error_result("All models failed")

            # Ensemble: average probabilities
            ai_probs = [r["ai_probability"] for r in model_results]
            avg_ai_prob = np.mean(ai_probs)
            std_ai_prob = np.std(ai_probs) if len(ai_probs) > 1 else 0

            # Agreement affects confidence
            agreement = 1 - std_ai_prob
            detected = avg_ai_prob > 0.5

            if agreement > 0.8 and avg_ai_prob > 0.7:
                confidence = Confidence.VERY_HIGH
            elif agreement > 0.6 and avg_ai_prob > 0.6:
                confidence = Confidence.HIGH
            elif agreement > 0.4:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            if detected:
                explanation = (
                    f"AI-generated image detected with {avg_ai_prob:.1%} probability "
                    f"({len(model_results)} models, agreement={agreement:.1%})"
                )
            else:
                explanation = (
                    f"Image appears authentic ({1-avg_ai_prob:.1%} real probability, "
                    f"{len(model_results)} models)"
                )

            limitations = [
                "ML models can be fooled by adversarial examples",
                "Performance varies by AI generator type",
                "Highly compressed images may affect accuracy",
            ]

            processing_time = (time.perf_counter() - start_time) * 1000

            return self._create_result(
                score=avg_ai_prob,
                confidence=confidence,
                detected=detected,
                explanation=explanation,
                limitations=limitations,
                details={
                    "model_results": model_results,
                    "average_ai_probability": float(avg_ai_prob),
                    "std_ai_probability": float(std_ai_prob),
                    "agreement": float(agreement),
                    "device": self._device,
                },
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"AI detection failed: {e}")
            return self._create_error_result(str(e))

    def _find_ai_index(self, id2label: dict) -> int:
        """Find which index corresponds to AI-generated."""
        for idx, label in id2label.items():
            label_lower = label.lower()
            if any(kw in label_lower for kw in ["ai", "artificial", "generated", "fake", "synthetic"]):
                return int(idx)
        return 1  # Default assumption


@register_plugin
class FrequencyAIDetector(ForensicsPlugin):
    """
    AI detection through frequency domain analysis.

    Detects AI-generated images through:
    - GAN fingerprints in FFT spectrum
    - Upsampling artifacts
    - Unnatural frequency distribution

    Works without ML models - CPU only, always available.
    """

    plugin_id = "frequency_ai_detector"
    plugin_name = "AI Detection (Frequency Analysis)"
    category = PluginCategory.AI_DETECTION
    description = "Detects AI fingerprints through frequency domain analysis"
    version = "1.0.0"

    def analyze(
        self,
        image: Image.Image,
        image_path: Optional[Path] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> ForensicsResult:
        """Detect AI through frequency analysis."""
        start_time = time.perf_counter()

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            img_array = np.array(image, dtype=np.float32)
            gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]

            # 2D FFT
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            log_magnitude = np.log1p(magnitude)

            h, w = gray.shape
            cy, cx = h // 2, w // 2

            # 1. Analyze radial frequency distribution
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Frequency bands
            low_mask = dist < min(h, w) * 0.1
            mid_mask = (dist >= min(h, w) * 0.1) & (dist < min(h, w) * 0.3)
            high_mask = dist >= min(h, w) * 0.3

            total_energy = np.sum(magnitude)
            low_energy = np.sum(magnitude[low_mask]) / total_energy
            mid_energy = np.sum(magnitude[mid_mask]) / total_energy
            high_energy = np.sum(magnitude[high_mask]) / total_energy

            # 2. Check for GAN-specific patterns
            # GANs often produce periodic artifacts
            # Calculate spectral flatness (AI images tend to have flatter spectrum)
            geometric_mean = np.exp(np.mean(np.log(magnitude[magnitude > 0] + 1e-10)))
            arithmetic_mean = np.mean(magnitude)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

            # 3. Look for periodic peaks (upsampling artifacts)
            peak_threshold = np.mean(log_magnitude) + 2.5 * np.std(log_magnitude)
            peaks = log_magnitude > peak_threshold
            peak_ratio = np.mean(peaks)

            # 4. High-frequency falloff analysis
            # Natural images have characteristic 1/f falloff
            # AI images often deviate from this
            high_freq_ratio = high_energy / (low_energy + 1e-10)

            # Score calculation
            ai_indicators = 0.0

            # Flatter spectrum = more AI-like
            if spectral_flatness > 0.25:
                ai_indicators += 0.3
            elif spectral_flatness > 0.15:
                ai_indicators += 0.15

            # More periodic peaks = more AI-like
            if peak_ratio > 0.01:
                ai_indicators += 0.25
            elif peak_ratio > 0.005:
                ai_indicators += 0.1

            # Unusual high-frequency content
            if high_freq_ratio < 0.3:
                ai_indicators += 0.2
            elif high_freq_ratio > 2.0:
                ai_indicators += 0.1  # Could indicate upscaling

            score = min(1.0, ai_indicators)
            detected = score > 0.4

            if score > 0.6:
                confidence = Confidence.MEDIUM
            elif score > 0.4:
                confidence = Confidence.LOW
            else:
                confidence = Confidence.LOW

            if detected:
                explanation = (
                    f"Frequency analysis suggests possible AI generation: "
                    f"spectral flatness={spectral_flatness:.3f}, "
                    f"periodic peaks={peak_ratio:.4f}"
                )
            else:
                explanation = (
                    f"Frequency characteristics consistent with natural image "
                    f"(flatness={spectral_flatness:.3f}, high_freq_ratio={high_freq_ratio:.2f})"
                )

            limitations = [
                "Frequency analysis is a weak signal alone",
                "Works best combined with ML-based detection",
                "Heavy processing can alter frequency characteristics",
            ]

            # Create frequency visualization
            freq_vis = self._create_frequency_visual(log_magnitude)

            processing_time = (time.perf_counter() - start_time) * 1000

            return self._create_result(
                score=score,
                confidence=confidence,
                detected=detected,
                explanation=explanation,
                limitations=limitations,
                details={
                    "spectral_flatness": float(spectral_flatness),
                    "peak_ratio": float(peak_ratio),
                    "low_energy": float(low_energy),
                    "mid_energy": float(mid_energy),
                    "high_energy": float(high_energy),
                    "high_freq_ratio": float(high_freq_ratio),
                },
                artifacts=[
                    Artifact(
                        name="frequency_spectrum",
                        artifact_type="heatmap",
                        data=freq_vis,
                        description="FFT magnitude spectrum - periodic patterns may indicate AI",
                    ),
                ],
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Frequency AI detection failed: {e}")
            return self._create_error_result(str(e))

    def _create_frequency_visual(self, log_magnitude: np.ndarray) -> Image.Image:
        """Create frequency spectrum visualization."""
        # Normalize to 0-255
        min_val, max_val = log_magnitude.min(), log_magnitude.max()
        normalized = (log_magnitude - min_val) / (max_val - min_val + 1e-10)
        img_array = (normalized * 255).astype(np.uint8)

        return Image.fromarray(img_array)
