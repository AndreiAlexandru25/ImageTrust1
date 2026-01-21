"""
Main AI Image Detection Pipeline.

Orchestrates preprocessing, model inference, and probability calibration.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from imagetrust.core.config import get_settings, Settings
from imagetrust.core.exceptions import ModelLoadingError, InvalidImageError
from imagetrust.core.types import DetectionScore, Confidence, DetectionVerdict
from imagetrust.detection.preprocessing import ImagePreprocessor
from imagetrust.utils.logging import get_logger
from imagetrust.utils.helpers import timer, load_image

logger = get_logger(__name__)


class AIDetector:
    """
    Main class for AI-generated image detection.
    
    Handles model loading, preprocessing, inference, and calibration.
    Can operate with a single model or an ensemble.
    
    Example:
        >>> detector = AIDetector(model="ensemble")
        >>> result = detector.detect("image.jpg")
        >>> print(f"AI Probability: {result['ai_probability']:.1%}")
    """

    def __init__(
        self,
        model: str = "ensemble",
        device: Optional[Union[str, torch.device]] = None,
        checkpoint: Optional[Path] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize the AI Detector.
        
        Args:
            model: Model name ("ensemble", "efficientnet_b4", "convnext_base", etc.)
            device: Computation device (auto-detect if None)
            checkpoint: Path to custom model checkpoint
            settings: Custom settings object
        """
        self.settings = settings or get_settings()
        self.model_name = model
        self.checkpoint = checkpoint
        
        # Determine device
        if device is None:
            self.device = self.settings.get_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Components
        self._model = None
        self._preprocessor = None
        self._calibrator = None
        
        # Initialize
        self._initialize()
        
        logger.info(f"AIDetector initialized: model={model}, device={self.device}")

    def _initialize(self) -> None:
        """Initialize model, preprocessor, and calibrator."""
        self._load_model()
        self._setup_preprocessor()
        self._setup_calibrator()

    def _load_model(self) -> None:
        """Load the detection model."""
        from imagetrust.detection.models.cnn_detector import CNNDetector
        from imagetrust.detection.models.vit_detector import ViTDetector
        from imagetrust.detection.models.ensemble import EnsembleDetector
        from imagetrust.detection.models.hf_detector import HuggingFaceDetector
        
        # Check if using a HuggingFace pretrained model (RECOMMENDED)
        hf_models = ["ai-detector", "sdxl-detector", "aiornot", "hf-", "umm-maybe", "Organika"]
        is_hf_model = any(hf in self.model_name for hf in hf_models)
        
        if is_hf_model or self.model_name == "ensemble":
            # Use HuggingFace pretrained model for best accuracy
            if self.model_name == "ensemble":
                # For ensemble, use the best HF model
                self._model = HuggingFaceDetector(
                    model_name="ai-detector",
                    device=self.device,
                )
            else:
                self._model = HuggingFaceDetector(
                    model_name=self.model_name,
                    device=self.device,
                )
            self._use_hf = True
            logger.info(f"Using HuggingFace pretrained AI detector")
        else:
            self._use_hf = False
            # Load custom model
            if "vit" in self.model_name or "deit" in self.model_name or "swin" in self.model_name:
                self._model = ViTDetector(
                    backbone=self.model_name,
                    device=self.device,
                    checkpoint=self.checkpoint,
                )
            else:
                self._model = CNNDetector(
                    backbone=self.model_name,
                    device=self.device,
                    checkpoint=self.checkpoint,
                )
            self._model.to(self.device)
            self._model.eval()

    def _setup_preprocessor(self) -> None:
        """Set up image preprocessor."""
        input_size = getattr(self._model, "input_size", self.settings.input_size)
        self._preprocessor = ImagePreprocessor(
            input_size=input_size,
            mean=self.settings.normalize_mean,
            std=self.settings.normalize_std,
        )

    def _setup_calibrator(self) -> None:
        """Set up probability calibrator."""
        from imagetrust.detection.calibration import CalibrationWrapper
        
        self._calibrator = CalibrationWrapper(
            model=self._model,
            calibration_method="temperature",
            min_confidence=self.settings.min_confidence,
            max_confidence=self.settings.max_confidence,
        )

    @property
    def model(self):
        """Access the underlying model."""
        return self._model

    @property
    def preprocessor(self):
        """Access the preprocessor."""
        return self._preprocessor

    def detect(
        self,
        image: Union[Image.Image, Path, str, bytes],
        use_calibration: bool = True,
    ) -> Dict[str, Any]:
        """
        Detect if an image is AI-generated.
        
        Args:
            image: Image to analyze (PIL Image, path, or bytes)
            use_calibration: Whether to apply probability calibration
            
        Returns:
            Dictionary with detection results
        """
        if self._model is None:
            raise ModelLoadingError("Model not loaded")
        
        # Load image
        if not isinstance(image, Image.Image):
            image = load_image(image)
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            if image.mode == "RGBA":
                bg = Image.new("RGB", image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[3])
                image = bg
            else:
                image = image.convert("RGB")
        
        with timer() as t:
            # Use HuggingFace model directly if available
            if getattr(self, "_use_hf", False):
                ai_prob, real_prob = self._model.predict_image(image)
            else:
                # Standard preprocessing pipeline
                tensor = self._preprocessor.preprocess(image)
                tensor = tensor.unsqueeze(0).to(self.device)
                
                # Inference
                with torch.no_grad():
                    if use_calibration and self._calibrator:
                        probs = self._calibrator(tensor)
                    else:
                        logits = self._model(tensor)
                        probs = torch.softmax(logits, dim=1)
                    
                    ai_prob = probs[0, 1].item()
                    real_prob = probs[0, 0].item()
        
        # Determine verdict and confidence
        verdict = DetectionVerdict.AI_GENERATED if ai_prob > 0.5 else DetectionVerdict.REAL
        if 0.4 <= ai_prob <= 0.6:
            verdict = DetectionVerdict.UNCERTAIN
        
        confidence = Confidence.from_probability(ai_prob)
        
        return {
            "ai_probability": ai_prob,
            "real_probability": real_prob,
            "verdict": verdict,
            "confidence": confidence,
            "calibrated": use_calibration,
            "processing_time_ms": t["elapsed_ms"],
            "model_name": self.model_name,
        }

    def detect_batch(
        self,
        images: List[Union[Image.Image, Path, str]],
        batch_size: int = 8,
        use_calibration: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Detect AI generation for a batch of images.
        
        Args:
            images: List of images to analyze
            batch_size: Processing batch size
            use_calibration: Whether to apply calibration
            
        Returns:
            List of detection results
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Load and preprocess
            pil_images = []
            for img in batch_images:
                if not isinstance(img, Image.Image):
                    img = load_image(img)
                pil_images.append(img)
            
            batch_tensor = self._preprocessor.preprocess_batch(pil_images)
            batch_tensor = batch_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                if use_calibration and self._calibrator:
                    probs = self._calibrator(batch_tensor)
                else:
                    logits = self._model(batch_tensor)
                    probs = torch.softmax(logits, dim=1)
            
            # Process results
            for j in range(len(batch_images)):
                ai_prob = probs[j, 1].item()
                real_prob = probs[j, 0].item()
                
                verdict = DetectionVerdict.AI_GENERATED if ai_prob > 0.5 else DetectionVerdict.REAL
                if 0.4 <= ai_prob <= 0.6:
                    verdict = DetectionVerdict.UNCERTAIN
                
                results.append({
                    "ai_probability": ai_prob,
                    "real_probability": real_prob,
                    "verdict": verdict,
                    "confidence": Confidence.from_probability(ai_prob),
                    "calibrated": use_calibration,
                    "model_name": self.model_name,
                })
        
        return results

    def predict(self, image: Union[Image.Image, Path, str]) -> Dict[str, Any]:
        """Alias for detect()."""
        return self.detect(image)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "input_size": self._preprocessor.input_size if self._preprocessor else None,
            "is_ensemble": self.model_name == "ensemble",
            "calibrated": self._calibrator is not None,
        }

    def to(self, device: Union[str, torch.device]) -> "AIDetector":
        """Move detector to a different device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if self._model:
            self._model.to(device)
        return self
