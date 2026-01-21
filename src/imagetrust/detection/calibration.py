"""
Probability calibration methods for AI detection.

Provides Temperature Scaling, Platt Scaling, and Isotonic Regression
to ensure reliable confidence scores.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from torch.optim import LBFGS

from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for calibration.
    
    A simple post-hoc calibration method that learns a single
    temperature parameter to scale logits.
    """

    def __init__(self, initial_temperature: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor([initial_temperature], dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature.clamp(min=0.01)

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
    ) -> None:
        """
        Learn optimal temperature from validation data.
        
        Args:
            logits: Model logits (N, C)
            labels: Ground truth labels (N,)
            max_iter: Maximum optimization iterations
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        logger.info(f"Temperature calibrated to: {self.temperature.item():.4f}")


class PlattScaling(nn.Module):
    """
    Platt Scaling for calibration.
    
    Fits a logistic regression on top of model outputs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calibrator = None
        self._fitted = False

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Fit Platt scaling using scikit-learn.
        
        Args:
            logits: Model logits or probabilities (N, C)
            labels: Ground truth labels (N,)
        """
        # Use probability of positive class
        if logits.dim() == 2:
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        else:
            probs = logits.cpu().numpy()
        
        labels_np = labels.cpu().numpy()
        
        self.calibrator = LogisticRegression()
        self.calibrator.fit(probs.reshape(-1, 1), labels_np)
        self._fitted = True
        
        logger.info("Platt scaling fitted")

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Platt scaling."""
        if not self._fitted:
            return torch.softmax(logits, dim=1)
        
        if logits.dim() == 2:
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        else:
            probs = logits.cpu().numpy()
        
        calibrated = self.calibrator.predict_proba(probs.reshape(-1, 1))
        return torch.tensor(calibrated, device=logits.device, dtype=logits.dtype)


class IsotonicCalibration(nn.Module):
    """
    Isotonic Regression calibration.
    
    Non-parametric calibration using isotonic regression.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calibrator = None
        self._fitted = False

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Fit isotonic regression.
        
        Args:
            logits: Model logits or probabilities (N, C)
            labels: Ground truth labels (N,)
        """
        if logits.dim() == 2:
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        else:
            probs = logits.cpu().numpy()
        
        labels_np = labels.cpu().numpy()
        
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(probs, labels_np)
        self._fitted = True
        
        logger.info("Isotonic calibration fitted")

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply isotonic calibration."""
        if not self._fitted:
            return torch.softmax(logits, dim=1)
        
        if logits.dim() == 2:
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        else:
            probs = logits.cpu().numpy()
        
        calibrated = self.calibrator.predict(probs)
        # Return as 2-class probabilities
        result = torch.zeros((len(calibrated), 2), device=logits.device)
        result[:, 1] = torch.tensor(calibrated, device=logits.device)
        result[:, 0] = 1 - result[:, 1]
        return result


class CalibrationWrapper(nn.Module):
    """
    Wrapper that applies calibration and confidence bounding to a model.
    
    Args:
        model: Base detection model
        calibration_method: "temperature", "platt", or "isotonic"
        min_confidence: Minimum allowed confidence
        max_confidence: Maximum allowed confidence
    """

    def __init__(
        self,
        model: nn.Module,
        calibration_method: str = "temperature",
        min_confidence: float = 0.80,
        max_confidence: float = 0.95,
    ) -> None:
        super().__init__()
        self.model = model
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        
        # Initialize calibrator
        if calibration_method == "temperature":
            self.calibrator = TemperatureScaling()
        elif calibration_method == "platt":
            self.calibrator = PlattScaling()
        elif calibration_method == "isotonic":
            self.calibrator = IsotonicCalibration()
        else:
            raise ValueError(f"Unknown calibration method: {calibration_method}")
        
        self._is_calibrated = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with calibration and confidence bounding."""
        # Get model output
        logits = self.model(x)
        
        # Apply calibration
        if self._is_calibrated:
            probs = self.calibrator(logits)
        else:
            probs = torch.softmax(logits, dim=1)
        
        # Bound confidence
        probs = self._bound_confidence(probs)
        
        return probs

    def _bound_confidence(self, probs: torch.Tensor) -> torch.Tensor:
        """Bound probabilities to [min_confidence, max_confidence] range."""
        # Get the maximum probability (confidence)
        max_prob = probs.max(dim=1, keepdim=True)[0]
        
        # Only bound if outside range
        scale = torch.ones_like(max_prob)
        
        # If max prob > max_confidence, scale down
        too_high = max_prob > self.max_confidence
        scale = torch.where(
            too_high,
            self.max_confidence / max_prob,
            scale
        )
        
        # Apply scaling (this maintains relative proportions)
        bounded = probs * scale
        
        # Renormalize to sum to 1
        bounded = bounded / bounded.sum(dim=1, keepdim=True)
        
        return bounded

    def calibrate(
        self,
        val_loader,
        device: torch.device,
    ) -> None:
        """
        Calibrate using validation data.
        
        Args:
            val_loader: DataLoader with (images, labels)
            device: Computation device
        """
        self.model.eval()
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = self.model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        self.calibrator.calibrate(all_logits, all_labels)
        self._is_calibrated = True
        
        logger.info("Model calibrated successfully")

    def predict_with_confidence(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions with confidence scores.
        
        Returns:
            Tuple of (predictions, confidence_scores, probabilities)
        """
        probs = self.forward(x)
        confidence, predictions = probs.max(dim=1)
        return predictions, confidence, probs


class ExpectedCalibrationError:
    """
    Computes Expected Calibration Error (ECE) and related metrics.
    """

    def __init__(self, n_bins: int = 15) -> None:
        self.n_bins = n_bins

    def compute(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute ECE and calibration statistics.
        
        Args:
            probs: Predicted probabilities (N, C) or (N,) for binary
            labels: True labels (N,)
            
        Returns:
            Tuple of (ECE value, statistics dict)
        """
        # Handle 2D probabilities
        if probs.ndim == 2:
            # Use probability of predicted class
            predictions = probs.argmax(axis=1)
            confidences = probs.max(axis=1)
            accuracies = (predictions == labels).astype(float)
        else:
            confidences = np.maximum(probs, 1 - probs)
            predictions = (probs > 0.5).astype(int)
            accuracies = (predictions == labels).astype(float)
        
        # Compute ECE
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                acc_in_bin = accuracies[in_bin].mean()
                conf_in_bin = confidences[in_bin].mean()
                ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
                
                bin_accuracies.append(acc_in_bin)
                bin_confidences.append(conf_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        # Maximum Calibration Error
        mce = max(
            np.abs(acc - conf)
            for acc, conf in zip(bin_accuracies, bin_confidences)
            if conf > 0
        ) if any(bin_counts) else 0.0
        
        stats = {
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
            "mce": mce,
        }
        
        return ece, stats
