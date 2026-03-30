"""
Meta-Classifier for Feature-Level Ensemble.

Implements learned combination of backbone embeddings + image quality features
for robust AI detection. Provides both XGBoost and MLP implementations for
ablation comparison.

Key components:
- EmbeddingExtractor: Extract penultimate layer features from any backbone
- NIQEComputer: No-Reference Image Quality Assessment
- MetaClassifier: Trainable feature-level ensemble (XGBoost or MLP)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


# Type aliases
Tensor = torch.Tensor
NDArray = np.ndarray


class BackboneType(Enum):
    """Supported backbone architectures."""
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B3 = "efficientnet_b3"
    EFFICIENTNET_B4 = "efficientnet_b4"
    VIT_B_16 = "vit_b_16"
    VIT_B_32 = "vit_b_32"
    VIT_L_16 = "vit_l_16"
    CONVNEXT_TINY = "convnext_tiny"
    CONVNEXT_SMALL = "convnext_small"
    CONVNEXT_BASE = "convnext_base"


# Embedding dimensions for each backbone
EMBEDDING_DIMS: Dict[BackboneType, int] = {
    BackboneType.RESNET50: 2048,
    BackboneType.RESNET101: 2048,
    BackboneType.EFFICIENTNET_B0: 1280,
    BackboneType.EFFICIENTNET_B3: 1536,
    BackboneType.EFFICIENTNET_B4: 1792,
    BackboneType.VIT_B_16: 768,
    BackboneType.VIT_B_32: 768,
    BackboneType.VIT_L_16: 1024,
    BackboneType.CONVNEXT_TINY: 768,
    BackboneType.CONVNEXT_SMALL: 768,
    BackboneType.CONVNEXT_BASE: 1024,
}


@dataclass
class EmbeddingResult:
    """Result from embedding extraction."""
    embeddings: Tensor  # Shape: (batch_size, embed_dim)
    backbone_name: str
    embed_dim: int


@dataclass
class QualityFeatures:
    """Image quality features for meta-classifier."""
    niqe_score: float  # No-Reference Image Quality (lower = better)
    brisque_score: float  # BRISQUE score (lower = better)
    sharpness: float  # Laplacian variance
    noise_level: float  # Estimated noise
    jpeg_quality_estimate: float  # Estimated JPEG quality
    features_vector: NDArray  # Concatenated feature vector


@dataclass
class MetaClassifierPrediction:
    """Prediction from meta-classifier."""
    ai_probability: float
    confidence: float
    is_uncertain: bool
    feature_importances: Optional[Dict[str, float]]
    raw_logit: float


class EmbeddingExtractor:
    """
    Extract penultimate layer features from any backbone.

    Uses forward hooks for efficient extraction without modifying model forward pass.
    Supports: ResNet, EfficientNet, ViT, ConvNeXt architectures.
    """

    def __init__(
        self,
        backbone: Union[nn.Module, str],
        backbone_type: Optional[BackboneType] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize embedding extractor.

        Args:
            backbone: Pre-trained backbone model or model name string.
            backbone_type: Type of backbone (for dimension lookup).
            device: Device to run on ("cuda" or "cpu").
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(backbone, str):
            self.backbone = self._load_backbone(backbone)
            self.backbone_type = BackboneType(backbone)
        else:
            self.backbone = backbone
            self.backbone_type = backbone_type

        self.backbone.to(self.device)
        self.backbone.eval()

        # Storage for extracted features
        self._features: Optional[Tensor] = None
        self._hook_handle = None

        # Register forward hook
        self._register_hook()

        # Get embedding dimension
        if self.backbone_type and self.backbone_type in EMBEDDING_DIMS:
            self.embed_dim = EMBEDDING_DIMS[self.backbone_type]
        else:
            # Infer from model architecture
            self.embed_dim = self._infer_embed_dim()

        logger.info(
            f"EmbeddingExtractor initialized: {self.backbone_type}, dim={self.embed_dim}"
        )

    def _load_backbone(self, model_name: str) -> nn.Module:
        """Load a pre-trained backbone model."""
        try:
            import timm
            model = timm.create_model(model_name, pretrained=True, num_classes=0)
            return model
        except ImportError:
            # Fallback to torchvision
            import torchvision.models as models
            if model_name == "resnet50":
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            elif model_name == "efficientnet_b0":
                model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            elif model_name == "vit_b_16":
                model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            return model

    def _register_hook(self):
        """Register forward hook on penultimate layer."""
        # Find the appropriate layer based on architecture
        target_layer = None

        # Check for common architectures
        if hasattr(self.backbone, "fc"):
            # ResNet-like
            target_layer = self.backbone.avgpool if hasattr(self.backbone, "avgpool") else None
        elif hasattr(self.backbone, "classifier"):
            # EfficientNet-like
            if hasattr(self.backbone, "avgpool"):
                target_layer = self.backbone.avgpool
            elif hasattr(self.backbone, "global_pool"):
                target_layer = self.backbone.global_pool
        elif hasattr(self.backbone, "heads"):
            # ViT-like
            if hasattr(self.backbone, "norm"):
                target_layer = self.backbone.norm
        elif hasattr(self.backbone, "head"):
            # ConvNeXt-like
            if hasattr(self.backbone, "norm"):
                target_layer = self.backbone.norm
            elif hasattr(self.backbone, "avgpool"):
                target_layer = self.backbone.avgpool

        # If timm model with forward_features
        if target_layer is None and hasattr(self.backbone, "forward_features"):
            # For timm models, we'll extract differently
            self._use_forward_features = True
            return

        self._use_forward_features = False

        if target_layer is None:
            # Fallback: use the layer before the last one
            children = list(self.backbone.children())
            if len(children) > 1:
                target_layer = children[-2]
            else:
                logger.warning("Could not find penultimate layer, using full model output")
                return

        def hook_fn(module, input, output):
            # Flatten if needed
            if isinstance(output, Tensor):
                if output.dim() > 2:
                    self._features = output.flatten(start_dim=1)
                else:
                    self._features = output
            else:
                self._features = output

        self._hook_handle = target_layer.register_forward_hook(hook_fn)

    def _infer_embed_dim(self) -> int:
        """Infer embedding dimension by running a dummy forward pass."""
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        with torch.no_grad():
            if hasattr(self, "_use_forward_features") and self._use_forward_features:
                features = self.backbone.forward_features(dummy_input)
                if features.dim() > 2:
                    features = features.flatten(start_dim=1)
                return features.shape[-1]
            else:
                _ = self.backbone(dummy_input)
                if self._features is not None:
                    return self._features.shape[-1]
                else:
                    return 512  # Default fallback

    def extract(
        self,
        images: Union[Tensor, List[Image.Image]],
        preprocess: bool = True,
    ) -> EmbeddingResult:
        """
        Extract embeddings from images.

        Args:
            images: Input images as tensor (B, C, H, W) or list of PIL images.
            preprocess: Whether to preprocess PIL images.

        Returns:
            EmbeddingResult with extracted features.
        """
        # Handle PIL images
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            from torchvision import transforms

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            tensors = [transform(img.convert("RGB")) for img in images]
            images = torch.stack(tensors)

        images = images.to(self.device)

        with torch.no_grad():
            if hasattr(self, "_use_forward_features") and self._use_forward_features:
                features = self.backbone.forward_features(images)
                if features.dim() > 2:
                    # Global average pooling for spatial features
                    if features.dim() == 4:
                        features = features.mean(dim=[2, 3])
                    elif features.dim() == 3:
                        features = features[:, 0]  # Use CLS token for ViT
                embeddings = features
            else:
                _ = self.backbone(images)
                embeddings = self._features

        if embeddings is None:
            raise RuntimeError("Failed to extract embeddings")

        return EmbeddingResult(
            embeddings=embeddings,
            backbone_name=str(self.backbone_type) if self.backbone_type else "unknown",
            embed_dim=embeddings.shape[-1],
        )

    def __call__(self, images: Union[Tensor, List[Image.Image]]) -> Tensor:
        """Convenience method for extraction."""
        return self.extract(images).embeddings

    def __del__(self):
        """Clean up hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()


class NIQEComputer:
    """
    No-Reference Image Quality Assessment.

    Computes various no-reference quality metrics:
    - NIQE (Naturalness Image Quality Evaluator)
    - BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
    - Sharpness (Laplacian variance)
    - Noise estimation
    - JPEG quality estimation

    These features help the meta-classifier distinguish between:
    - Real photos with natural degradation
    - AI images with synthetic perfection
    - Screenshots with rendering artifacts
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize quality computer.

        Args:
            use_gpu: Whether to use GPU for computation.
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"

        # Try to import piq for quality metrics
        self._piq_available = False
        try:
            import piq
            if hasattr(piq, "NIQE") and hasattr(piq, "BRISQUE"):
                self._piq_available = True
                self._niqe = piq.NIQE()
                self._brisque = piq.BRISQUE()
                logger.info("PIQ library available for quality assessment")
            else:
                logger.warning("PIQ library found but NIQE/BRISQUE not available. Using fallback.")
        except Exception:
            logger.warning(
                "PIQ library not available. Using fallback quality metrics. "
                "Install with: pip install piq"
            )

    def _compute_sharpness(self, image: NDArray) -> float:
        """Compute sharpness using Laplacian variance."""
        import cv2

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _compute_noise_level(self, image: NDArray) -> float:
        """Estimate noise level using MAD estimator."""
        from scipy import ndimage

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # High-pass filter to isolate noise
        laplacian = ndimage.laplace(gray.astype(np.float64))

        # MAD estimator for noise
        sigma = np.median(np.abs(laplacian)) / 0.6745
        return float(sigma)

    def _estimate_jpeg_quality(self, image: NDArray) -> float:
        """Estimate JPEG quality from blocking artifacts."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        h, w = gray.shape

        # Compute block boundary differences (8x8 blocks typical for JPEG)
        block_size = 8

        # Horizontal block boundaries
        h_boundaries = []
        for i in range(block_size, h, block_size):
            diff = np.abs(gray[i, :] - gray[i - 1, :])
            h_boundaries.append(np.mean(diff))

        # Vertical block boundaries
        v_boundaries = []
        for j in range(block_size, w, block_size):
            diff = np.abs(gray[:, j] - gray[:, j - 1])
            v_boundaries.append(np.mean(diff))

        # Average boundary artifact strength
        if h_boundaries and v_boundaries:
            artifact_strength = (np.mean(h_boundaries) + np.mean(v_boundaries)) / 2
        else:
            artifact_strength = 0.0

        # Map artifact strength to quality estimate (inverse relationship)
        # Higher artifacts = lower quality
        # This is a rough heuristic
        estimated_quality = max(0, min(100, 100 - artifact_strength * 2))
        return float(estimated_quality)

    def _fallback_niqe(self, image: NDArray) -> float:
        """Fallback NIQE approximation using statistical features."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Compute local mean subtracted contrast normalized (MSCN) coefficients
        from scipy.ndimage import gaussian_filter

        mu = gaussian_filter(gray, sigma=7 / 6)
        sigma = np.sqrt(gaussian_filter((gray - mu) ** 2, sigma=7 / 6))
        sigma = np.maximum(sigma, 1e-6)

        mscn = (gray - mu) / sigma

        # Compute generalized Gaussian distribution parameters
        mean_mscn = np.mean(mscn)
        var_mscn = np.var(mscn)
        skew_mscn = np.mean((mscn - mean_mscn) ** 3) / (var_mscn ** 1.5 + 1e-10)

        # NIQE-like score (higher = worse quality)
        # This is a simplified version
        niqe_approx = np.abs(mean_mscn) + np.abs(1 - var_mscn) + np.abs(skew_mscn)
        return float(niqe_approx * 10)  # Scale to similar range as NIQE

    def _fallback_brisque(self, image: NDArray) -> float:
        """Fallback BRISQUE approximation."""
        # Use similar approach to NIQE fallback with additional features
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        sharpness = self._compute_sharpness(image)
        noise = self._compute_noise_level(image)

        # BRISQUE-like score combining multiple factors
        brisque_approx = (100 - sharpness / 100) + noise * 5
        return float(np.clip(brisque_approx, 0, 100))

    def compute(self, image: Union[Image.Image, NDArray, Tensor]) -> QualityFeatures:
        """
        Compute quality features for an image.

        Args:
            image: Input image (PIL, numpy, or tensor).

        Returns:
            QualityFeatures with all computed metrics.
        """
        # Convert to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert("RGB"))
        elif isinstance(image, Tensor):
            img_array = image.cpu().numpy()
            if img_array.shape[0] == 3:  # CHW format
                img_array = np.transpose(img_array, (1, 2, 0))
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = image.astype(np.uint8) if image.max() > 1.0 else (image * 255).astype(np.uint8)

        # Compute NIQE and BRISQUE
        if self._piq_available:
            # Convert to tensor for PIQ
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_tensor = img_tensor.to(self.device)

            try:
                niqe_score = self._niqe(img_tensor).item()
            except Exception:
                niqe_score = self._fallback_niqe(img_array)

            try:
                brisque_score = self._brisque(img_tensor).item()
            except Exception:
                brisque_score = self._fallback_brisque(img_array)
        else:
            niqe_score = self._fallback_niqe(img_array)
            brisque_score = self._fallback_brisque(img_array)

        # Compute other features
        sharpness = self._compute_sharpness(img_array)
        noise_level = self._compute_noise_level(img_array)
        jpeg_quality = self._estimate_jpeg_quality(img_array)

        # Create feature vector
        features_vector = np.array([
            niqe_score,
            brisque_score,
            sharpness,
            noise_level,
            jpeg_quality,
        ], dtype=np.float32)

        return QualityFeatures(
            niqe_score=niqe_score,
            brisque_score=brisque_score,
            sharpness=sharpness,
            noise_level=noise_level,
            jpeg_quality_estimate=jpeg_quality,
            features_vector=features_vector,
        )

    def compute_batch(
        self,
        images: List[Union[Image.Image, NDArray]],
    ) -> List[QualityFeatures]:
        """Compute quality features for a batch of images."""
        return [self.compute(img) for img in images]


class BaseMetaClassifier(ABC):
    """Abstract base class for meta-classifiers."""

    @abstractmethod
    def fit(
        self,
        embeddings: NDArray,
        quality_features: NDArray,
        labels: NDArray,
    ) -> None:
        """Train the meta-classifier."""
        pass

    @abstractmethod
    def predict_proba(
        self,
        embeddings: NDArray,
        quality_features: NDArray,
    ) -> NDArray:
        """Predict probabilities."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""
        pass


class XGBoostMetaClassifier(BaseMetaClassifier):
    """
    XGBoost-based meta-classifier.

    Advantages:
    - Fast training and inference
    - Built-in feature importance
    - Handles heterogeneous features well
    - Good with smaller datasets

    Use case: When embeddings are frozen and you want interpretability.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize XGBoost meta-classifier.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate.
            random_state: Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model = None
        self.feature_names: List[str] = []

        # Try to import xgboost
        try:
            import xgboost as xgb
            self._xgb = xgb
            self._available = True
        except ImportError:
            self._available = False
            logger.warning(
                "XGBoost not available. Install with: pip install xgboost"
            )

    def fit(
        self,
        embeddings: NDArray,
        quality_features: NDArray,
        labels: NDArray,
        embed_feature_names: Optional[List[str]] = None,
        quality_feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Train the XGBoost meta-classifier.

        Args:
            embeddings: Backbone embeddings (N, embed_dim).
            quality_features: Quality features (N, n_quality_features).
            labels: Binary labels (N,).
            embed_feature_names: Optional names for embedding dimensions.
            quality_feature_names: Optional names for quality features.
        """
        if not self._available:
            raise RuntimeError("XGBoost is not available")

        # Concatenate features
        X = np.hstack([embeddings, quality_features])

        # Create feature names
        embed_dim = embeddings.shape[1]
        if embed_feature_names is None:
            embed_feature_names = [f"emb_{i}" for i in range(embed_dim)]
        if quality_feature_names is None:
            quality_feature_names = ["niqe", "brisque", "sharpness", "noise", "jpeg_q"]

        self.feature_names = embed_feature_names + quality_feature_names

        # Create DMatrix
        dtrain = self._xgb.DMatrix(X, label=labels, feature_names=self.feature_names)

        # XGBoost parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc"],
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "seed": self.random_state,
            "verbosity": 1,
        }

        # Train
        self.model = self._xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False,
        )

        logger.info(f"XGBoost meta-classifier trained with {X.shape[1]} features")

    def predict_proba(
        self,
        embeddings: NDArray,
        quality_features: NDArray,
    ) -> NDArray:
        """Predict probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X = np.hstack([embeddings, quality_features])
        dtest = self._xgb.DMatrix(X, feature_names=self.feature_names)

        probs = self.model.predict(dtest)
        return probs

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}

        importance = self.model.get_score(importance_type="gain")
        return importance

    def save(self, path: Path) -> None:
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")

        path = Path(path)
        self.model.save_model(str(path.with_suffix(".xgb")))

        # Save feature names
        import json
        with open(path.with_suffix(".json"), "w") as f:
            json.dump({"feature_names": self.feature_names}, f)

    def load(self, path: Path) -> None:
        """Load model from disk."""
        if not self._available:
            raise RuntimeError("XGBoost is not available")

        path = Path(path)
        self.model = self._xgb.Booster()
        self.model.load_model(str(path.with_suffix(".xgb")))

        # Load feature names
        import json
        with open(path.with_suffix(".json")) as f:
            data = json.load(f)
            self.feature_names = data["feature_names"]


class MLPMetaClassifier(BaseMetaClassifier, nn.Module):
    """
    MLP-based meta-classifier.

    Advantages:
    - End-to-end trainable with backbones
    - Higher capacity for complex patterns
    - GPU acceleration

    Use case: When fine-tuning backbones or need maximum performance.
    """

    def __init__(
        self,
        embed_dim: int,
        quality_dim: int = 5,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        device: Optional[str] = None,
    ):
        """
        Initialize MLP meta-classifier.

        Args:
            embed_dim: Dimension of backbone embeddings.
            quality_dim: Dimension of quality features.
            hidden_dims: Hidden layer dimensions.
            dropout: Dropout rate.
            device: Device to run on.
        """
        nn.Module.__init__(self)
        BaseMetaClassifier.__init__(self)

        self.embed_dim = embed_dim
        self.quality_dim = quality_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Input dimension = embeddings + quality features
        input_dim = embed_dim + quality_dim

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)
        self.to(self.device)

        logger.info(
            f"MLP meta-classifier initialized: input={input_dim}, "
            f"hidden={hidden_dims}, output=1"
        )

    def forward(
        self,
        embeddings: Tensor,
        quality_features: Tensor,
    ) -> Tensor:
        """Forward pass."""
        x = torch.cat([embeddings, quality_features], dim=1)
        return self.mlp(x)

    def fit(
        self,
        embeddings: NDArray,
        quality_features: NDArray,
        labels: NDArray,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        val_split: float = 0.1,
    ) -> Dict[str, List[float]]:
        """
        Train the MLP meta-classifier.

        Args:
            embeddings: Backbone embeddings (N, embed_dim).
            quality_features: Quality features (N, quality_dim).
            labels: Binary labels (N,).
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            weight_decay: L2 regularization weight.
            val_split: Validation split fraction.

        Returns:
            Training history dict with losses.
        """
        from torch.utils.data import DataLoader, TensorDataset, random_split

        # Convert to tensors
        X_emb = torch.from_numpy(embeddings).float().to(self.device)
        X_qual = torch.from_numpy(quality_features).float().to(self.device)
        y = torch.from_numpy(labels).float().to(self.device)

        # Create dataset
        dataset = TensorDataset(X_emb, X_qual, y)

        # Split into train/val
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )

        history = {"train_loss": [], "val_loss": [], "val_auc": []}

        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0

            for emb_batch, qual_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = self.forward(emb_batch, qual_batch).squeeze()
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation
            self.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for emb_batch, qual_batch, y_batch in val_loader:
                    logits = self.forward(emb_batch, qual_batch).squeeze()
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()

                    probs = torch.sigmoid(logits)
                    val_preds.extend(probs.cpu().numpy())
                    val_labels.extend(y_batch.cpu().numpy())

            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)

            # Compute AUC
            from sklearn.metrics import roc_auc_score
            if len(np.unique(val_labels)) > 1:
                val_auc = roc_auc_score(val_labels, val_preds)
            else:
                val_auc = 0.5
            history["val_auc"].append(val_auc)

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"val_auc={val_auc:.4f}"
                )

        return history

    def predict_proba(
        self,
        embeddings: NDArray,
        quality_features: NDArray,
    ) -> NDArray:
        """Predict probabilities."""
        self.eval()

        X_emb = torch.from_numpy(embeddings).float().to(self.device)
        X_qual = torch.from_numpy(quality_features).float().to(self.device)

        with torch.no_grad():
            logits = self.forward(X_emb, X_qual).squeeze()
            probs = torch.sigmoid(logits)

        return probs.cpu().numpy()

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        torch.save({
            "state_dict": self.state_dict(),
            "embed_dim": self.embed_dim,
            "quality_dim": self.quality_dim,
        }, path.with_suffix(".pt"))

    def load(self, path: Path) -> None:
        """Load model from disk."""
        path = Path(path)
        checkpoint = torch.load(path.with_suffix(".pt"), map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])


class MetaClassifier:
    """
    Unified meta-classifier interface.

    Combines multiple backbone embeddings with quality features using
    either XGBoost (fast, interpretable) or MLP (end-to-end trainable).

    Usage:
        meta_clf = MetaClassifier(
            backbone_names=["resnet50", "efficientnet_b0", "vit_b_16"],
            classifier_type="xgboost",
        )
        meta_clf.fit(images, labels)
        predictions = meta_clf.predict(new_images)
    """

    def __init__(
        self,
        backbone_names: Optional[List[str]] = None,
        classifier_type: str = "xgboost",  # "xgboost" or "mlp"
        device: Optional[str] = None,
        **classifier_kwargs,
    ):
        """
        Initialize meta-classifier.

        Args:
            backbone_names: List of backbone model names for embedding extraction.
            classifier_type: Type of classifier ("xgboost" or "mlp").
            device: Device to run on.
            **classifier_kwargs: Additional arguments for the classifier.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Default backbones
        if backbone_names is None:
            backbone_names = ["resnet50", "efficientnet_b0"]

        self.backbone_names = backbone_names

        # Initialize embedding extractors (lazy loading)
        self._extractors: Dict[str, EmbeddingExtractor] = {}

        # Initialize quality computer
        self.quality_computer = NIQEComputer(use_gpu=self.device == "cuda")

        # Initialize classifier
        self.classifier_type = classifier_type
        self.classifier_kwargs = classifier_kwargs
        self._classifier: Optional[BaseMetaClassifier] = None

        # Total embedding dimension
        self._total_embed_dim: Optional[int] = None

        logger.info(
            f"MetaClassifier initialized with backbones: {backbone_names}, "
            f"classifier: {classifier_type}"
        )

    def _get_extractor(self, backbone_name: str) -> EmbeddingExtractor:
        """Get or create embedding extractor for a backbone."""
        if backbone_name not in self._extractors:
            self._extractors[backbone_name] = EmbeddingExtractor(
                backbone=backbone_name,
                device=self.device,
            )
        return self._extractors[backbone_name]

    def _extract_all_embeddings(
        self,
        images: List[Image.Image],
    ) -> Tuple[NDArray, NDArray]:
        """Extract embeddings from all backbones and compute quality features."""
        all_embeddings = []

        for backbone_name in self.backbone_names:
            extractor = self._get_extractor(backbone_name)
            result = extractor.extract(images)
            all_embeddings.append(result.embeddings.cpu().numpy())

        # Concatenate all embeddings
        embeddings = np.hstack(all_embeddings)
        self._total_embed_dim = embeddings.shape[1]

        # Compute quality features
        quality_results = self.quality_computer.compute_batch(images)
        quality_features = np.stack([q.features_vector for q in quality_results])

        return embeddings, quality_features

    def fit(
        self,
        images: List[Image.Image],
        labels: NDArray,
        **fit_kwargs,
    ) -> Dict[str, Any]:
        """
        Train the meta-classifier.

        Args:
            images: List of training images.
            labels: Binary labels (0=real, 1=AI).
            **fit_kwargs: Additional arguments for classifier fit().

        Returns:
            Training info/history dict.
        """
        logger.info(f"Extracting features from {len(images)} images...")

        # Extract features
        embeddings, quality_features = self._extract_all_embeddings(images)

        logger.info(
            f"Features extracted: embeddings={embeddings.shape}, "
            f"quality={quality_features.shape}"
        )

        # Initialize classifier
        if self.classifier_type == "xgboost":
            self._classifier = XGBoostMetaClassifier(**self.classifier_kwargs)
        elif self.classifier_type == "mlp":
            self._classifier = MLPMetaClassifier(
                embed_dim=self._total_embed_dim,
                quality_dim=quality_features.shape[1],
                **self.classifier_kwargs,
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

        # Train
        result = self._classifier.fit(embeddings, quality_features, labels, **fit_kwargs)

        return result if isinstance(result, dict) else {"trained": True}

    def predict(
        self,
        images: List[Image.Image],
    ) -> List[MetaClassifierPrediction]:
        """
        Make predictions on images.

        Args:
            images: List of images to classify.

        Returns:
            List of MetaClassifierPrediction objects.
        """
        if self._classifier is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Extract features
        embeddings, quality_features = self._extract_all_embeddings(images)

        # Predict
        probs = self._classifier.predict_proba(embeddings, quality_features)

        # Get feature importance if available
        feature_importance = None
        if hasattr(self._classifier, "get_feature_importance"):
            feature_importance = self._classifier.get_feature_importance()

        # Build predictions
        predictions = []
        for i, prob in enumerate(probs):
            predictions.append(MetaClassifierPrediction(
                ai_probability=float(prob),
                confidence=float(abs(prob - 0.5) * 2),  # Scale to 0-1
                is_uncertain=0.3 < prob < 0.7,
                feature_importances=feature_importance,
                raw_logit=float(np.log(prob / (1 - prob + 1e-10))),
            ))

        return predictions

    def save(self, path: Union[str, Path]) -> None:
        """Save meta-classifier to disk."""
        if self._classifier is None:
            raise RuntimeError("No model to save")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save classifier
        self._classifier.save(path / f"meta_classifier_{self.classifier_type}")

        # Save config
        import json
        config = {
            "backbone_names": self.backbone_names,
            "classifier_type": self.classifier_type,
            "total_embed_dim": self._total_embed_dim,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Meta-classifier saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load meta-classifier from disk."""
        path = Path(path)

        # Load config
        import json
        with open(path / "config.json") as f:
            config = json.load(f)

        self.backbone_names = config["backbone_names"]
        self.classifier_type = config["classifier_type"]
        self._total_embed_dim = config["total_embed_dim"]

        # Load classifier
        if self.classifier_type == "xgboost":
            self._classifier = XGBoostMetaClassifier(**self.classifier_kwargs)
        elif self.classifier_type == "mlp":
            self._classifier = MLPMetaClassifier(
                embed_dim=self._total_embed_dim,
                quality_dim=5,
                **self.classifier_kwargs,
            )

        self._classifier.load(path / f"meta_classifier_{self.classifier_type}")

        logger.info(f"Meta-classifier loaded from {path}")
