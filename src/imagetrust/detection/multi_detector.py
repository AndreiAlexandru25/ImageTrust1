"""
Multi-Model AI Detection System.

Combines multiple detection methods for robust AI image detection:
1. Multiple HuggingFace models (ensemble voting)
2. Frequency analysis (FFT artifacts)
3. Texture analysis (noise patterns)
4. Edge coherence analysis
5. Color distribution analysis
6. Meta-classifier for feature-level ensemble (NEW)
7. Conformal prediction for rigorous uncertainty (NEW)
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
import torch
from dataclasses import dataclass
from enum import Enum

from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)

# Check for optional components
META_CLASSIFIER_AVAILABLE = False
CONFORMAL_AVAILABLE = False

try:
    from imagetrust.detection.meta_classifier import MetaClassifier, MetaClassifierPrediction
    META_CLASSIFIER_AVAILABLE = True
except ImportError:
    pass

try:
    from imagetrust.detection.conformal import ConformalPredictor, ConformalPrediction
    CONFORMAL_AVAILABLE = True
except ImportError:
    pass


class AnalysisType(Enum):
    """Types of analysis performed."""
    ML_MODEL = "ml_model"
    FREQUENCY = "frequency"
    TEXTURE = "texture"
    EDGES = "edges"
    COLOR = "color"
    NOISE = "noise"


@dataclass
class DetectionResult:
    """Result from a single detection method."""
    method: str
    ai_probability: float
    confidence: float
    details: Dict[str, Any]
    weight: float = 1.0


class FrequencyAnalyzer:
    """
    Analyze image in frequency domain to detect AI artifacts.
    
    AI-generated images often have:
    - Different high-frequency patterns
    - Periodic artifacts from upsampling
    - Unusual spectral distribution
    """
    
    def analyze(self, image: Image.Image) -> DetectionResult:
        """Analyze frequency domain characteristics."""
        # Convert to grayscale numpy
        gray = np.array(image.convert("L"), dtype=np.float32)
        
        # Apply FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Log magnitude spectrum
        log_magnitude = np.log1p(magnitude)
        
        # Analyze different frequency bands
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        
        # Create frequency band masks
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Low, mid, high frequency energy
        low_mask = dist < min(h, w) * 0.1
        mid_mask = (dist >= min(h, w) * 0.1) & (dist < min(h, w) * 0.3)
        high_mask = dist >= min(h, w) * 0.3
        
        total_energy = np.sum(magnitude)
        low_energy = np.sum(magnitude[low_mask]) / total_energy
        mid_energy = np.sum(magnitude[mid_mask]) / total_energy
        high_energy = np.sum(magnitude[high_mask]) / total_energy
        
        # AI images often have less natural high-frequency content
        # or unusual periodic patterns
        high_freq_ratio = high_energy / (low_energy + 1e-10)
        
        # Check for periodic artifacts (common in GAN upsampling)
        # Look for peaks in the spectrum
        peak_threshold = np.mean(log_magnitude) + 2 * np.std(log_magnitude)
        num_peaks = np.sum(log_magnitude > peak_threshold)
        peak_ratio = num_peaks / (h * w)
        
        # Compute spectral flatness (real images usually have more varied spectrum)
        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
        arithmetic_mean = np.mean(magnitude)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        # Score calculation
        # Higher spectral flatness = more AI-like
        # Lower high_freq_ratio = more AI-like
        # Higher peak_ratio = more AI-like (periodic artifacts)
        
        ai_score = 0.0
        
        # Spectral flatness indicator
        if spectral_flatness > 0.3:
            ai_score += 0.3
        elif spectral_flatness > 0.2:
            ai_score += 0.15
        
        # High frequency analysis
        if high_freq_ratio < 0.5:
            ai_score += 0.25
        elif high_freq_ratio < 1.0:
            ai_score += 0.1
        
        # Periodic artifacts
        if peak_ratio > 0.01:
            ai_score += 0.2
        elif peak_ratio > 0.005:
            ai_score += 0.1
        
        # Normalize
        ai_probability = min(1.0, ai_score / 0.75)
        
        return DetectionResult(
            method="Frequency Analysis (FFT)",
            ai_probability=ai_probability,
            confidence=0.4,  # Lower confidence - supporting evidence only
            details={
                "spectral_flatness": float(spectral_flatness),
                "high_freq_ratio": float(high_freq_ratio),
                "peak_ratio": float(peak_ratio),
                "low_energy": float(low_energy),
                "mid_energy": float(mid_energy),
                "high_energy": float(high_energy),
            },
            weight=0.05  # Low weight - ML models are primary
        )


class TextureAnalyzer:
    """
    Analyze texture patterns for AI detection.
    
    AI images often have:
    - More uniform noise patterns
    - Less natural texture variation
    - Unusual local statistics
    """
    
    def analyze(self, image: Image.Image) -> DetectionResult:
        """Analyze texture characteristics."""
        gray = np.array(image.convert("L"), dtype=np.float32)
        
        # Local Binary Pattern-like analysis
        # Calculate local variance in patches
        patch_size = 16
        h, w = gray.shape
        
        variances = []
        means = []
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = gray[i:i+patch_size, j:j+patch_size]
                variances.append(np.var(patch))
                means.append(np.mean(patch))
        
        variances = np.array(variances)
        means = np.array(means)
        
        # Statistics of local statistics
        var_of_var = np.var(variances)
        mean_var = np.mean(variances)
        
        # Coefficient of variation of local variances
        cv_variance = np.std(variances) / (np.mean(variances) + 1e-10)
        
        # AI images often have more uniform texture
        # Real images have higher variation in local statistics
        
        ai_score = 0.0
        
        # Low variation in local variance = AI-like
        if cv_variance < 0.5:
            ai_score += 0.4
        elif cv_variance < 1.0:
            ai_score += 0.2
        
        # Very uniform mean values = AI-like  
        mean_cv = np.std(means) / (np.mean(means) + 1e-10)
        if mean_cv < 0.3:
            ai_score += 0.3
        elif mean_cv < 0.5:
            ai_score += 0.15
        
        ai_probability = min(1.0, ai_score / 0.7)
        
        return DetectionResult(
            method="Texture Analysis",
            ai_probability=ai_probability,
            confidence=0.35,
            details={
                "variance_cv": float(cv_variance),
                "mean_cv": float(mean_cv),
                "mean_local_variance": float(mean_var),
                "var_of_variance": float(var_of_var),
            },
            weight=0.04  # Low weight
        )


class NoiseAnalyzer:
    """
    Analyze noise patterns for AI detection.
    
    AI images have different noise characteristics than camera sensors.
    """
    
    def analyze(self, image: Image.Image) -> DetectionResult:
        """Analyze noise patterns."""
        img_array = np.array(image.convert("RGB"), dtype=np.float32)
        
        # Estimate noise using Laplacian
        from scipy import ndimage
        
        noise_estimates = []
        for channel in range(3):
            # Laplacian for edge detection
            laplacian = ndimage.laplace(img_array[:, :, channel])
            
            # Noise estimate using MAD (Median Absolute Deviation)
            sigma = np.median(np.abs(laplacian)) / 0.6745
            noise_estimates.append(sigma)
        
        noise_estimates = np.array(noise_estimates)
        
        # Noise level consistency across channels
        noise_std = np.std(noise_estimates)
        noise_mean = np.mean(noise_estimates)
        noise_cv = noise_std / (noise_mean + 1e-10)
        
        # AI images often have very consistent noise across channels
        # Real camera noise varies more between channels
        
        ai_score = 0.0
        
        # Very consistent noise = AI-like
        if noise_cv < 0.1:
            ai_score += 0.4
        elif noise_cv < 0.2:
            ai_score += 0.2
        
        # Very low noise = possibly AI (or very processed)
        if noise_mean < 1.0:
            ai_score += 0.3
        elif noise_mean < 3.0:
            ai_score += 0.15
        
        ai_probability = min(1.0, ai_score / 0.7)
        
        return DetectionResult(
            method="Noise Pattern Analysis",
            ai_probability=ai_probability,
            confidence=0.4,
            details={
                "noise_mean": float(noise_mean),
                "noise_std": float(noise_std),
                "noise_cv": float(noise_cv),
                "channel_noise": noise_estimates.tolist(),
            },
            weight=0.04
        )


class EdgeCoherenceAnalyzer:
    """
    Analyze edge coherence for AI detection.
    
    AI images often have:
    - Unnatural edge patterns
    - Inconsistent edge strength
    - Artifacts at object boundaries
    """
    
    def analyze(self, image: Image.Image) -> DetectionResult:
        """Analyze edge coherence."""
        from scipy import ndimage
        
        gray = np.array(image.convert("L"), dtype=np.float32)
        
        # Sobel edge detection
        sobel_x = ndimage.sobel(gray, axis=1)
        sobel_y = ndimage.sobel(gray, axis=0)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Edge direction
        edge_direction = np.arctan2(sobel_y, sobel_x)
        
        # Analyze edge statistics
        edge_mean = np.mean(edge_magnitude)
        edge_std = np.std(edge_magnitude)
        
        # Gradient direction histogram
        # Real images have more varied edge directions
        direction_hist, _ = np.histogram(edge_direction.flatten(), bins=36, range=(-np.pi, np.pi))
        direction_hist = direction_hist / (np.sum(direction_hist) + 1e-10)
        direction_entropy = -np.sum(direction_hist * np.log(direction_hist + 1e-10))
        max_entropy = np.log(36)
        normalized_entropy = direction_entropy / max_entropy
        
        # Edge magnitude distribution
        # AI images often have bimodal distribution (strong edges or nothing)
        edge_flat = edge_magnitude.flatten()
        percentile_25 = np.percentile(edge_flat, 25)
        percentile_75 = np.percentile(edge_flat, 75)
        iqr = percentile_75 - percentile_25
        edge_spread = iqr / (edge_mean + 1e-10)
        
        ai_score = 0.0
        
        # Low direction entropy = AI-like (less varied edges)
        if normalized_entropy < 0.7:
            ai_score += 0.35
        elif normalized_entropy < 0.85:
            ai_score += 0.15
        
        # Low edge spread = AI-like
        if edge_spread < 1.0:
            ai_score += 0.25
        elif edge_spread < 1.5:
            ai_score += 0.1
        
        ai_probability = min(1.0, ai_score / 0.6)
        
        return DetectionResult(
            method="Edge Coherence Analysis",
            ai_probability=ai_probability,
            confidence=0.35,
            details={
                "direction_entropy": float(normalized_entropy),
                "edge_spread": float(edge_spread),
                "edge_mean": float(edge_mean),
                "edge_std": float(edge_std),
            },
            weight=0.04
        )


class ColorAnalyzer:
    """
    Analyze color distribution for AI detection.
    """
    
    def analyze(self, image: Image.Image) -> DetectionResult:
        """Analyze color characteristics."""
        img_array = np.array(image.convert("RGB"), dtype=np.float32)
        
        # Color channel correlations
        r, g, b = img_array[:,:,0].flatten(), img_array[:,:,1].flatten(), img_array[:,:,2].flatten()
        
        # Correlation between channels
        rg_corr = np.corrcoef(r, g)[0, 1]
        rb_corr = np.corrcoef(r, b)[0, 1]
        gb_corr = np.corrcoef(g, b)[0, 1]
        
        mean_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3
        
        # Color histogram analysis
        color_hist = []
        for channel in range(3):
            hist, _ = np.histogram(img_array[:,:,channel].flatten(), bins=32, range=(0, 256))
            hist = hist / (np.sum(hist) + 1e-10)
            color_hist.append(hist)
        
        # Entropy of color distribution
        entropies = []
        for hist in color_hist:
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            entropies.append(entropy)
        
        mean_entropy = np.mean(entropies)
        max_entropy = np.log(32)
        normalized_color_entropy = mean_entropy / max_entropy
        
        ai_score = 0.0
        
        # Very high channel correlation = AI-like
        if mean_corr > 0.95:
            ai_score += 0.3
        elif mean_corr > 0.9:
            ai_score += 0.15
        
        # Low color entropy = AI-like (less color variation)
        if normalized_color_entropy < 0.6:
            ai_score += 0.25
        elif normalized_color_entropy < 0.75:
            ai_score += 0.1
        
        ai_probability = min(1.0, ai_score / 0.55)
        
        return DetectionResult(
            method="Color Distribution Analysis",
            ai_probability=ai_probability,
            confidence=0.35,
            details={
                "channel_correlation": float(mean_corr),
                "color_entropy": float(normalized_color_entropy),
                "rg_corr": float(rg_corr),
                "rb_corr": float(rb_corr),
                "gb_corr": float(gb_corr),
            },
            weight=0.03
        )


class MultiModelDetector:
    """
    Combines multiple HuggingFace AI detection models.
    Uses the BEST available pretrained models for maximum accuracy.
    
    Top models by downloads and accuracy (January 2025):
    - dima806/deepfake_vs_real: 26k downloads, best for general AI
    - Organika/sdxl-detector: 8k downloads, best for SDXL/SD
    - dima806/ai_vs_real: 3.6k downloads, alternative
    """
    
    # TOP MODELS - TESTED AND VERIFIED TO WORK CORRECTLY
    # Ensemble of 4 models for better accuracy through voting
    AVAILABLE_MODELS = [
        # #1 - Best overall accuracy for general images (26k downloads)
        ("dima806/deepfake_vs_real_image_detection", "Deepfake vs Real"),
        # #2 - Good general detector
        ("umm-maybe/AI-image-detector", "AI Image Detector"),
        # #3 - Alternative with different architecture
        ("Nahrawy/AIorNot", "AIorNot Detector"),
        # #4 - NEW: Recent model from NYUAD (Jan 2025)
        ("NYUAD-ComNets/NYUAD_AI-generated_images_detector", "NYUAD Detector (2025)"),
    ]
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.processors = {}
        self._load_models()
    
    def _load_models(self):
        """Load all available models."""
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        
        for model_id, name in self.AVAILABLE_MODELS:
            try:
                logger.info(f"Loading model: {name}")
                processor = AutoImageProcessor.from_pretrained(model_id)
                model = AutoModelForImageClassification.from_pretrained(model_id)
                model.to(self.device)
                model.eval()
                
                self.models[model_id] = {
                    "model": model,
                    "processor": processor,
                    "name": name,
                    "id2label": model.config.id2label,
                }
                logger.info(f"Loaded: {name}")
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
    
    def _compute_ai_probability(
        self, probs: torch.Tensor, id2label: dict
    ) -> float:
        """
        Compute P(AI-generated) from model output probabilities.

        Handles different label conventions:
        - Binary:  {0: 'Real', 1: 'Fake'}
        - Binary:  {0: 'artificial', 1: 'human'}
        - Binary:  {0: 'real', 1: 'ai'}
        - Multi-class: {0: 'dalle', 1: 'real', 2: 'sd'}  (NYUAD)

        For multi-class: sums all non-real classes as P(AI).
        """
        # Keywords indicating AI/synthetic content
        ai_keywords = [
            "ai", "artificial", "generated", "fake", "synthetic",
            "dalle", "dall-e", "midjourney", "sd", "stable",
            "stylegan", "progan", "biggan", "glide", "deepfake",
        ]
        # Keywords indicating real/authentic content
        real_keywords = ["real", "human", "natural", "authentic", "photo"]

        # Identify AI indices and real indices
        ai_indices = []
        real_indices = []

        for idx, label in id2label.items():
            idx_int = int(idx)
            label_lower = label.lower().strip()
            is_ai = any(kw in label_lower for kw in ai_keywords)
            is_real = any(kw in label_lower for kw in real_keywords)

            if is_ai and not is_real:
                ai_indices.append(idx_int)
            elif is_real and not is_ai:
                real_indices.append(idx_int)
            elif is_ai and is_real:
                # Ambiguous, treat as AI
                ai_indices.append(idx_int)
            # else: unrecognized label, skip

        if ai_indices:
            # Sum all AI-class probabilities
            return sum(probs[i].item() for i in ai_indices)
        elif real_indices:
            # No AI labels found but real labels found -> P(AI) = 1 - P(real)
            real_sum = sum(probs[i].item() for i in real_indices)
            return 1.0 - real_sum
        else:
            # Fallback: assume index 1 is AI
            logger.warning(
                f"Could not determine AI class from labels: {id2label}. "
                f"Defaulting to index 1."
            )
            return probs[1].item() if probs.shape[0] > 1 else 0.5

    def analyze(self, image: Image.Image) -> List[DetectionResult]:
        """Run all models on the image."""
        results = []

        for model_id, model_info in self.models.items():
            try:
                processor = model_info["processor"]
                model = model_info["model"]
                name = model_info["name"]

                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)[0]

                ai_prob = self._compute_ai_probability(
                    probs, model_info["id2label"]
                )

                results.append(DetectionResult(
                    method=f"ML: {name}",
                    ai_probability=ai_prob,
                    confidence=0.90,
                    details={
                        "model_id": model_id,
                        "all_probs": probs.cpu().numpy().tolist(),
                        "labels": model_info["id2label"],
                        "ai_probability_computed": ai_prob,
                    },
                    weight=0.40
                ))
            except Exception as e:
                logger.error(f"Error with {model_id}: {e}")

        return results


class ComprehensiveDetector:
    """
    Comprehensive AI Image Detection System.

    Combines all detection methods for maximum accuracy.
    Supports optional custom-trained Kaggle model for highest accuracy.

    Supports ablation study parameters to enable/disable components.
    Integrates uncertainty estimation for UNCERTAIN verdict support.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        kaggle_model_path: Optional[str] = None,
        # Ablation parameters
        enable_frequency: bool = True,
        enable_noise: bool = True,
        enable_texture: bool = True,
        enable_edges: bool = True,
        enable_color: bool = True,
        active_ml_models: Optional[List[str]] = None,
        ensemble_strategy: str = "weighted",
        # Meta-classifier parameters (NEW)
        use_meta_classifier: bool = False,
        meta_classifier_path: Optional[str] = None,
        meta_classifier_type: str = "xgboost",
        # Conformal prediction parameters (NEW)
        use_conformal: bool = False,
        conformal_alpha: float = 0.1,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Ablation configuration
        self.enable_frequency = enable_frequency
        self.enable_noise = enable_noise
        self.enable_texture = enable_texture
        self.enable_edges = enable_edges
        self.enable_color = enable_color
        self.active_ml_models = active_ml_models  # None = all models
        self.ensemble_strategy_name = ensemble_strategy

        # Meta-classifier and conformal configuration
        self.use_meta_classifier = use_meta_classifier
        self.use_conformal = use_conformal
        self.conformal_alpha = conformal_alpha

        # Initialize all analyzers
        self.ml_detector = MultiModelDetector(device=self.device)
        self.frequency_analyzer = FrequencyAnalyzer()
        self.texture_analyzer = TextureAnalyzer()
        self.noise_analyzer = NoiseAnalyzer()
        self.edge_analyzer = EdgeCoherenceAnalyzer()
        self.color_analyzer = ColorAnalyzer()

        # Load custom Kaggle-trained model if available
        self.kaggle_detector = None
        self._load_kaggle_model(kaggle_model_path)

        # Calibrated CNN ensemble (3 models: ResNet50, EfficientNet, ConvNeXt)
        self.calibrated_ensemble = None
        self._load_calibrated_ensemble()

        # Uncertainty estimator (lazy loaded)
        self._uncertainty_estimator = None

        # Meta-classifier (NEW)
        self.meta_classifier = None
        if use_meta_classifier and META_CLASSIFIER_AVAILABLE:
            self._load_meta_classifier(meta_classifier_path, meta_classifier_type)

        # Conformal predictor (NEW)
        self.conformal_predictor = None
        if use_conformal and CONFORMAL_AVAILABLE:
            self._init_conformal_predictor(conformal_alpha)

        logger.info("ComprehensiveDetector initialized with all analyzers")

    def _load_calibrated_ensemble(self) -> None:
        """Try to load all 3 calibrated CNN models as ensemble."""
        try:
            from imagetrust.detection.models.calibrated_ensemble import (
                CalibratedCNNEnsemble,
            )
            ensemble = CalibratedCNNEnsemble(
                device=self.device,
                strategy="min",
            )
            if ensemble.models:
                self.calibrated_ensemble = ensemble
                logger.info(
                    f"Calibrated CNN ensemble loaded: "
                    f"{len(ensemble.models)}/3 models"
                )
            else:
                logger.info("No calibrated CNN models found, skipping ensemble")
        except Exception as e:
            logger.warning(f"Could not load calibrated ensemble: {e}")

    def _load_kaggle_model(self, model_path: Optional[str] = None) -> None:
        """Try to load the Kaggle-trained deepfake detection model."""
        from pathlib import Path

        # Search paths for the model
        search_paths = []
        if model_path:
            search_paths.append(Path(model_path))

        # Default locations
        search_paths.extend([
            Path("models/best_model.pth"),
            Path("models/kaggle_deepfake.pth"),
            Path("models/swa_model.pth"),
        ])

        for path in search_paths:
            if path.exists():
                try:
                    from imagetrust.detection.models.kaggle_detector import (
                        load_kaggle_model,
                    )
                    self.kaggle_detector = load_kaggle_model(
                        checkpoint_path=path,
                        device=self.device,
                    )
                    logger.info(
                        f"Kaggle-trained model loaded from: {path}"
                    )
                    return
                except Exception as e:
                    logger.warning(
                        f"Failed to load Kaggle model from {path}: {e}"
                    )

        logger.info(
            "No Kaggle-trained model found. "
            "Place best_model.pth in models/ to enable."
        )

    def _load_meta_classifier(
        self,
        model_path: Optional[str] = None,
        classifier_type: str = "xgboost",
    ) -> None:
        """Load meta-classifier for feature-level ensemble."""
        if not META_CLASSIFIER_AVAILABLE:
            logger.warning("Meta-classifier not available")
            return

        try:
            from pathlib import Path

            # Search paths for meta-classifier
            search_paths = []
            if model_path:
                search_paths.append(Path(model_path))

            search_paths.extend([
                Path("models/meta_classifier"),
                Path("models/meta_classifier_xgboost"),
                Path("models/meta_classifier_mlp"),
            ])

            for path in search_paths:
                if path.exists():
                    try:
                        self.meta_classifier = MetaClassifier(
                            classifier_type=classifier_type,
                            device=self.device,
                        )
                        self.meta_classifier.load(path)
                        logger.info(f"Meta-classifier loaded from: {path}")
                        return
                    except Exception as e:
                        logger.warning(f"Failed to load meta-classifier from {path}: {e}")

            # Initialize fresh meta-classifier (needs training)
            self.meta_classifier = MetaClassifier(
                classifier_type=classifier_type,
                device=self.device,
            )
            logger.info(
                "Meta-classifier initialized (not trained). "
                "Call meta_classifier.fit() to train."
            )

        except Exception as e:
            logger.warning(f"Could not initialize meta-classifier: {e}")
            self.meta_classifier = None

    def _init_conformal_predictor(self, alpha: float = 0.1) -> None:
        """Initialize conformal predictor for rigorous uncertainty."""
        if not CONFORMAL_AVAILABLE:
            logger.warning("Conformal prediction not available")
            return

        try:
            from imagetrust.detection.conformal import ConformalPredictor, ConformalMethod

            self.conformal_predictor = ConformalPredictor(
                alpha=alpha,
                method=ConformalMethod.APS,
                labels=("real", "ai_generated"),
            )
            logger.info(
                f"Conformal predictor initialized with alpha={alpha} "
                f"(coverage={1-alpha:.0%})"
            )
            logger.info(
                "Note: Call calibrate_conformal() with calibration data before use"
            )

        except Exception as e:
            logger.warning(f"Could not initialize conformal predictor: {e}")
            self.conformal_predictor = None

    def calibrate_conformal(
        self,
        cal_images: List[Image.Image],
        cal_labels: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calibrate the conformal predictor using calibration data.

        Args:
            cal_images: List of calibration images.
            cal_labels: Binary labels (0=real, 1=AI).

        Returns:
            Calibration result dictionary.
        """
        if self.conformal_predictor is None:
            raise RuntimeError("Conformal predictor not initialized")

        # Get predictions on calibration set
        cal_probs = []
        for image in cal_images:
            result = self.analyze(
                image,
                return_uncertainty=False,
                profile=False,
            )
            cal_probs.append(result["ai_probability"])

        cal_probs = np.array(cal_probs)

        # Calibrate
        result = self.conformal_predictor.calibrate(cal_probs, cal_labels)

        return {
            "threshold": result.threshold,
            "coverage_level": result.coverage_level,
            "empirical_coverage": result.empirical_coverage,
            "avg_set_size": result.avg_set_size,
        }
    
    def analyze(
        self,
        image: Image.Image,
        return_uncertainty: bool = True,
        uncertainty_method: str = "entropy",
        abstain_threshold: Optional[float] = None,
        profile: bool = False,
    ) -> Dict[str, Any]:
        """
        Run comprehensive analysis on an image.

        Returns detailed results from all detection methods.

        NOTE: ML models are the PRIMARY detection method.
        Signal analysis is OPTIONAL and can cause false positives.

        Args:
            image: Input PIL Image
            return_uncertainty: Include uncertainty estimation in results
            uncertainty_method: Method for uncertainty (entropy, margin, confidence, ensemble)
            abstain_threshold: Threshold for abstaining (UNCERTAIN verdict)
            profile: Include timing breakdown for efficiency analysis

        Returns:
            Dictionary with detection results, uncertainty, and optional timing
        """
        import time

        timing = {} if profile else None

        # Ensure RGB
        if image.mode != "RGB":
            if image.mode == "RGBA":
                bg = Image.new("RGB", image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[3])
                image = bg
            else:
                image = image.convert("RGB")

        results: List[DetectionResult] = []

        # 0. Custom Kaggle-trained model (highest accuracy if available)
        if self.kaggle_detector is not None:
            if profile:
                t0 = time.perf_counter()
            try:
                ai_prob, real_prob = self.kaggle_detector.predict_image(image)
                results.append(DetectionResult(
                    method="ML: Custom Trained (Kaggle ResNet50, 97% acc)",
                    ai_probability=ai_prob,
                    confidence=0.95,
                    details={
                        "model_type": "kaggle_trained",
                        "backbone": self.kaggle_detector.backbone_name,
                        "ai_prob": ai_prob,
                        "real_prob": real_prob,
                    },
                    weight=0.50,
                ))
            except Exception as e:
                logger.error(f"Kaggle model inference failed: {e}")
            if profile:
                timing["kaggle_model_ms"] = (time.perf_counter() - t0) * 1000

        # 0b. Calibrated CNN Ensemble (3 custom-trained models with temp scaling)
        calibrated_prediction = None
        if self.calibrated_ensemble is not None:
            if profile:
                t0 = time.perf_counter()
            try:
                calibrated_prediction = self.calibrated_ensemble.predict(image)
                # Add each calibrated model as individual result
                for model_name, cal_prob in calibrated_prediction.calibrated_probs.items():
                    raw_prob = calibrated_prediction.raw_probs.get(model_name, cal_prob)
                    results.append(DetectionResult(
                        method=f"ML: CNN {model_name} (calibrated)",
                        ai_probability=cal_prob,
                        confidence=0.95,
                        details={
                            "model_type": "calibrated_cnn",
                            "backbone": model_name,
                            "raw_probability": raw_prob,
                            "calibrated_probability": cal_prob,
                            "temperature": self.calibrated_ensemble.thresholds.get(
                                model_name, {}
                            ).get("temperature", 1.0),
                        },
                        weight=0.45,
                    ))
                # Add ensemble result
                ens_prob = (
                    calibrated_prediction.ensemble_min_prob
                    if self.calibrated_ensemble.strategy == "min"
                    else calibrated_prediction.ensemble_avg_prob
                )
                results.append(DetectionResult(
                    method=f"ML: CNN Ensemble ({self.calibrated_ensemble.strategy})",
                    ai_probability=ens_prob,
                    confidence=0.97,
                    details={
                        "model_type": "calibrated_ensemble",
                        "strategy": self.calibrated_ensemble.strategy,
                        "ensemble_avg": calibrated_prediction.ensemble_avg_prob,
                        "ensemble_min": calibrated_prediction.ensemble_min_prob,
                        "ensemble_std": calibrated_prediction.ensemble_std,
                        "model_agreement": calibrated_prediction.model_agreement,
                        "verdict": calibrated_prediction.verdict,
                        "uncertain_region": [
                            calibrated_prediction.uncertain_low,
                            calibrated_prediction.uncertain_high,
                        ],
                    },
                    weight=0.55,
                ))
            except Exception as e:
                logger.error(f"Calibrated ensemble failed: {e}")
            if profile:
                timing["calibrated_ensemble_ms"] = (time.perf_counter() - t0) * 1000

        # 1. ML Models - PRIMARY DETECTION (these are trained specifically for AI detection)
        if profile:
            t0 = time.perf_counter()

        ml_results = self.ml_detector.analyze(image)

        # Filter ML results if active_ml_models is specified
        if self.active_ml_models is not None:
            ml_results = [
                r for r in ml_results
                if any(model_name in r.method for model_name in self.active_ml_models)
            ]

        results.extend(ml_results)

        if profile:
            timing["ml_models_ms"] = (time.perf_counter() - t0) * 1000

        # Signal analysis - provides UNIQUE forensic insights
        # These are scientific methods that complement ML models

        # 2. Frequency Analysis (FFT - detects upsampling artifacts)
        if self.enable_frequency:
            if profile:
                t0 = time.perf_counter()
            freq_result = self.frequency_analyzer.analyze(image)
            results.append(freq_result)
            if profile:
                timing["frequency_ms"] = (time.perf_counter() - t0) * 1000

        # 3. Noise Analysis (camera sensor fingerprint)
        if self.enable_noise:
            if profile:
                t0 = time.perf_counter()
            noise_result = self.noise_analyzer.analyze(image)
            results.append(noise_result)
            if profile:
                timing["noise_ms"] = (time.perf_counter() - t0) * 1000

        # 4. Texture Analysis (optional)
        if self.enable_texture:
            if profile:
                t0 = time.perf_counter()
            texture_result = self.texture_analyzer.analyze(image)
            results.append(texture_result)
            if profile:
                timing["texture_ms"] = (time.perf_counter() - t0) * 1000

        # 5. Edge Analysis (optional)
        if self.enable_edges:
            if profile:
                t0 = time.perf_counter()
            edge_result = self.edge_analyzer.analyze(image)
            results.append(edge_result)
            if profile:
                timing["edge_ms"] = (time.perf_counter() - t0) * 1000

        # 6. Color Analysis (optional)
        if self.enable_color:
            if profile:
                t0 = time.perf_counter()
            color_result = self.color_analyzer.analyze(image)
            results.append(color_result)
            if profile:
                timing["color_ms"] = (time.perf_counter() - t0) * 1000

        # Calculate weighted ensemble score
        if profile:
            t0 = time.perf_counter()

        total_weight = sum(r.weight * r.confidence for r in results)
        weighted_ai_prob = sum(r.ai_probability * r.weight * r.confidence for r in results) / total_weight

        # Voting analysis
        votes_ai = sum(1 for r in results if r.ai_probability > 0.5)
        votes_real = len(results) - votes_ai

        # Confidence based on agreement
        agreement = max(votes_ai, votes_real) / len(results)
        overall_confidence = agreement * 0.5 + 0.5  # Scale to 0.5-1.0

        if profile:
            timing["ensemble_ms"] = (time.perf_counter() - t0) * 1000

        # Uncertainty estimation
        uncertainty_info = None
        should_abstain = False

        if return_uncertainty:
            # Extract ML model probabilities for ensemble variance
            ml_probs = [r.ai_probability for r in results if r.method.startswith("ML:")]

            try:
                from imagetrust.baselines.uncertainty import UncertaintyEstimator

                if self._uncertainty_estimator is None:
                    self._uncertainty_estimator = UncertaintyEstimator(
                        method=uncertainty_method,
                        abstain_threshold=abstain_threshold,
                    )

                uncertainty_result = self._uncertainty_estimator.predict_with_uncertainty(
                    weighted_ai_prob,
                    ensemble_probs=ml_probs,
                )

                uncertainty_info = {
                    "score": uncertainty_result.uncertainty,
                    "method": uncertainty_result.uncertainty_method,
                    "should_abstain": uncertainty_result.should_abstain,
                    "confidence_level": uncertainty_result.confidence_level,
                    "ensemble_std": float(np.std(ml_probs)) if ml_probs else 0.0,
                }

                should_abstain = uncertainty_result.should_abstain

            except ImportError:
                # Fallback: simple entropy-based uncertainty
                eps = 1e-10
                p = np.clip(weighted_ai_prob, eps, 1 - eps)
                uncertainty = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
                should_abstain = uncertainty > (abstain_threshold or 0.5)

                uncertainty_info = {
                    "score": float(uncertainty),
                    "method": "entropy_fallback",
                    "should_abstain": should_abstain,
                    "confidence_level": "low" if uncertainty > 0.7 else "medium" if uncertainty > 0.4 else "high",
                    "ensemble_std": float(np.std(ml_probs)) if ml_probs else 0.0,
                }

        # Determine verdict
        if should_abstain:
            verdict = "uncertain"
            verdict_text = "Uncertain (High Uncertainty)"
        elif weighted_ai_prob > 0.6:
            verdict = "ai_generated"
            verdict_text = "AI-Generated"
        elif weighted_ai_prob < 0.4:
            verdict = "real"
            verdict_text = "Real Photograph"
        else:
            verdict = "uncertain"
            verdict_text = "Uncertain"

        # Build result dictionary
        result = {
            "ai_probability": weighted_ai_prob,
            "real_probability": 1 - weighted_ai_prob,
            "verdict": verdict,
            "verdict_text": verdict_text,
            "confidence": overall_confidence,
            "votes": {
                "ai": votes_ai,
                "real": votes_real,
                "total": len(results)
            },
            "individual_results": [
                {
                    "method": r.method,
                    "ai_probability": r.ai_probability,
                    "confidence": r.confidence,
                    "weight": r.weight,
                    "details": r.details,
                }
                for r in results
            ],
            "analysis_summary": {
                "ml_models": len(ml_results),
                "signal_analysis": sum([
                    self.enable_frequency,
                    self.enable_noise,
                    self.enable_texture,
                    self.enable_edges,
                ]),
                "color_analysis": 1 if self.enable_color else 0,
            }
        }

        # Add calibrated CNN ensemble results
        if calibrated_prediction is not None:
            result["calibrated_ensemble"] = {
                "raw_probs": calibrated_prediction.raw_probs,
                "calibrated_probs": calibrated_prediction.calibrated_probs,
                "ensemble_avg_prob": calibrated_prediction.ensemble_avg_prob,
                "ensemble_min_prob": calibrated_prediction.ensemble_min_prob,
                "verdict": calibrated_prediction.verdict,
                "verdict_text": calibrated_prediction.verdict_text,
                "strategy": calibrated_prediction.strategy_used,
                "uncertain_low": calibrated_prediction.uncertain_low,
                "uncertain_high": calibrated_prediction.uncertain_high,
                "ensemble_std": calibrated_prediction.ensemble_std,
                "model_agreement": calibrated_prediction.model_agreement,
            }

        # Add uncertainty info
        if uncertainty_info is not None:
            result["uncertainty"] = uncertainty_info

        # Add timing breakdown
        if timing is not None:
            timing["total_ms"] = sum(timing.values())
            result["timing_breakdown"] = timing

        # Meta-classifier prediction (NEW)
        if self.use_meta_classifier and self.meta_classifier is not None:
            try:
                meta_preds = self.meta_classifier.predict([image])
                if meta_preds:
                    meta_pred = meta_preds[0]
                    result["meta_classifier"] = {
                        "ai_probability": meta_pred.ai_probability,
                        "confidence": meta_pred.confidence,
                        "is_uncertain": meta_pred.is_uncertain,
                        "raw_logit": meta_pred.raw_logit,
                    }
                    if meta_pred.feature_importances:
                        result["meta_classifier"]["feature_importances"] = meta_pred.feature_importances
            except Exception as e:
                logger.warning(f"Meta-classifier prediction failed: {e}")

        # Conformal prediction (NEW)
        if self.use_conformal and self.conformal_predictor is not None:
            try:
                if self.conformal_predictor._calibrated:
                    conf_pred = self.conformal_predictor.predict(weighted_ai_prob)
                    result["conformal_prediction"] = {
                        "prediction_set": list(conf_pred.prediction_set),
                        "set_size": conf_pred.set_size,
                        "coverage_level": conf_pred.coverage_level,
                        "is_uncertain": conf_pred.is_uncertain,
                        "threshold": conf_pred.threshold,
                        "conformity_scores": conf_pred.conformity_scores,
                    }

                    # Override uncertainty if conformal says uncertain
                    if conf_pred.is_uncertain:
                        result["verdict"] = "uncertain"
                        result["verdict_text"] = "Uncertain (Conformal Prediction)"
                        if uncertainty_info is not None:
                            uncertainty_info["conformal_uncertain"] = True
                else:
                    result["conformal_prediction"] = {
                        "status": "not_calibrated",
                        "message": "Conformal predictor needs calibration",
                    }
            except Exception as e:
                logger.warning(f"Conformal prediction failed: {e}")

        return result
