"""
B1: Classical Baseline - LogReg/XGBoost on Forensic Features.

This baseline extracts handcrafted forensic features and trains a
simple classifier. It serves as a sanity check and lower bound.

For the paper, report:
- Feature extraction time per image
- Classifier type (LogReg or XGBoost)
- Regularization (C for LogReg, max_depth for XGBoost)
- Number of features used
- Training time
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from imagetrust.baselines.base import BaselineDetector, BaselineConfig, BaselineResult
from imagetrust.baselines.feature_extraction import ForensicFeatureExtractor, FeatureConfig


class ClassicalBaseline(BaselineDetector):
    """
    Classical baseline using handcrafted features + LogReg/XGBoost.

    Features extracted:
    - DCT coefficients (frequency domain)
    - Noise residual statistics
    - JPEG blocking artifacts
    - Color channel statistics
    - Local Binary Patterns (texture)
    - Edge statistics

    Example:
        >>> config = BaselineConfig(name="Classical (LogReg)")
        >>> baseline = ClassicalBaseline(config, classifier="logistic_regression")
        >>> baseline.fit(train_images, train_labels)
        >>> result = baseline.predict_proba(test_image)
    """

    def __init__(
        self,
        config: BaselineConfig,
        classifier: str = "logistic_regression",
        feature_config: Optional[FeatureConfig] = None,
        **classifier_kwargs,
    ):
        """
        Initialize classical baseline.

        Args:
            config: Baseline configuration
            classifier: "logistic_regression" or "xgboost"
            feature_config: Feature extraction configuration
            **classifier_kwargs: Additional classifier arguments
        """
        super().__init__(config)

        self.classifier_type = classifier
        self.classifier_kwargs = classifier_kwargs
        self.feature_extractor = ForensicFeatureExtractor(feature_config or FeatureConfig())

        self._classifier = None
        self._scaler = None

        # Store config for paper reporting
        self.config.model_params["classifier"] = classifier
        self.config.model_params["num_features"] = self.feature_extractor.num_features
        self.config.model_params.update(classifier_kwargs)

    def _create_classifier(self):
        """Create the classifier instance."""
        if self.classifier_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            defaults = {
                "C": 1.0,
                "max_iter": 1000,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "random_state": self.config.seed,
            }
            defaults.update(self.classifier_kwargs)
            return LogisticRegression(**defaults)

        elif self.classifier_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                defaults = {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": self.config.seed,
                    "use_label_encoder": False,
                    "eval_metric": "logloss",
                }
                defaults.update(self.classifier_kwargs)
                return XGBClassifier(**defaults)
            except ImportError:
                raise ImportError(
                    "XGBoost not installed. Install with: pip install xgboost"
                )
        else:
            raise ValueError(f"Unknown classifier: {self.classifier_type}")

    def _extract_features_for_dataset(
        self,
        images: List[Union[Image.Image, Path]],
        desc: str = "Extracting features",
        cache_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Extract features from a list of images.

        Supports caching for reproducibility and speed.
        """
        if cache_path and cache_path.exists():
            return np.load(cache_path)

        features = []
        for img in tqdm(images, desc=desc):
            pil_img = self._load_image(img)
            feat = self.feature_extractor.extract(pil_img)
            features.append(feat)

        features_array = np.stack(features)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, features_array)

        return features_array

    def fit(
        self,
        train_images: List[Union[Image.Image, Path]],
        train_labels: List[int],
        val_images: Optional[List[Union[Image.Image, Path]]] = None,
        val_labels: Optional[List[int]] = None,
        feature_cache_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Train the classical baseline.

        Args:
            train_images: Training images
            train_labels: Labels (0=real, 1=AI)
            val_images: Optional validation images
            val_labels: Optional validation labels
            feature_cache_dir: Optional directory to cache extracted features

        Returns:
            Training history dict
        """
        from sklearn.preprocessing import StandardScaler

        # Extract features
        cache_path = feature_cache_dir / "train_features.npy" if feature_cache_dir else None
        X_train = self._extract_features_for_dataset(
            train_images, "Extracting train features", cache_path
        )
        y_train = np.array(train_labels)

        # Normalize features
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)

        # Create and train classifier
        self._classifier = self._create_classifier()
        self._classifier.fit(X_train_scaled, y_train)

        self.is_fitted = True

        # Compute training metrics
        history = {
            "train_samples": len(train_images),
            "num_features": X_train.shape[1],
        }

        # Training accuracy
        train_probs = self._classifier.predict_proba(X_train_scaled)[:, 1]
        train_preds = (train_probs > 0.5).astype(int)
        history["train_accuracy"] = (train_preds == y_train).mean()

        # Validation metrics if provided
        if val_images is not None and val_labels is not None:
            cache_path = feature_cache_dir / "val_features.npy" if feature_cache_dir else None
            X_val = self._extract_features_for_dataset(
                val_images, "Extracting val features", cache_path
            )
            y_val = np.array(val_labels)
            X_val_scaled = self._scaler.transform(X_val)

            val_probs = self._classifier.predict_proba(X_val_scaled)[:, 1]
            val_preds = (val_probs > 0.5).astype(int)
            history["val_accuracy"] = (val_preds == y_val).mean()
            history["val_samples"] = len(val_images)

        return history

    def predict_proba(self, image: Union[Image.Image, np.ndarray, Path]) -> BaselineResult:
        """
        Predict AI probability for a single image.

        Args:
            image: Input image

        Returns:
            BaselineResult with predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Load and extract features
        pil_img = self._load_image(image)

        def _predict():
            features = self.feature_extractor.extract(pil_img)
            features_scaled = self._scaler.transform(features.reshape(1, -1))
            probs = self._classifier.predict_proba(features_scaled)[0]
            return probs, features

        (probs, features), elapsed_ms = self._timed_predict(_predict)

        # probs[0] = P(real), probs[1] = P(AI) for sklearn classifiers
        ai_prob = float(probs[1])
        real_prob = float(probs[0])

        # Apply calibration if available
        raw_prob = ai_prob
        if self._calibrator is not None:
            ai_prob = self._calibrator.calibrate(ai_prob)
            real_prob = 1 - ai_prob

        return BaselineResult(
            ai_probability=ai_prob,
            real_probability=real_prob,
            raw_probability=raw_prob,
            baseline_name=self.name,
            processing_time_ms=elapsed_ms,
            calibrated=self._calibrator is not None,
            features=features,
        )

    def predict_proba_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, Path]],
    ) -> List[BaselineResult]:
        """
        Batch prediction for efficiency.

        Args:
            images: List of images

        Returns:
            List of BaselineResult
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Extract all features
        features_list = []
        for img in images:
            pil_img = self._load_image(img)
            features_list.append(self.feature_extractor.extract(pil_img))

        features_array = np.stack(features_list)
        features_scaled = self._scaler.transform(features_array)

        # Batch prediction
        probs = self._classifier.predict_proba(features_scaled)

        # Build results
        results = []
        for i, (prob, feat) in enumerate(zip(probs, features_list)):
            ai_prob = float(prob[1])
            raw_prob = ai_prob

            if self._calibrator is not None:
                ai_prob = self._calibrator.calibrate(ai_prob)

            results.append(BaselineResult(
                ai_probability=ai_prob,
                real_probability=1 - ai_prob,
                raw_probability=raw_prob,
                baseline_name=self.name,
                processing_time_ms=0,  # Batch timing not per-image
                calibrated=self._calibrator is not None,
                features=feat,
            ))

        return results

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "classifier": self._classifier,
            "scaler": self._scaler,
            "config": self.config,
            "classifier_type": self.classifier_type,
            "classifier_kwargs": self.classifier_kwargs,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Union[str, Path]) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        self._classifier = state["classifier"]
        self._scaler = state["scaler"]
        self.config = state["config"]
        self.classifier_type = state["classifier_type"]
        self.classifier_kwargs = state["classifier_kwargs"]
        self.is_fitted = state["is_fitted"]

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (for XGBoost) or coefficients (for LogReg).

        Returns:
            Dict mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        feature_names = self.feature_extractor.feature_names

        if self.classifier_type == "xgboost":
            importance = self._classifier.feature_importances_
        else:
            # LogReg: use absolute coefficient values
            importance = np.abs(self._classifier.coef_[0])

        return dict(zip(feature_names, importance))
