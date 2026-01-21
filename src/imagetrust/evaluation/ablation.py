"""
Ablation study module.

Inspired by MLE-STAR methodology for systematic component analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from imagetrust.evaluation.metrics import compute_metrics
from imagetrust.utils.logging import get_logger
from imagetrust.utils.helpers import ensure_dir

logger = get_logger(__name__)


class AblationStudy:
    """
    Ablation study framework inspired by MLE-STAR.
    
    Systematically analyzes the contribution of each component
    to overall model performance.
    
    Components analyzed:
    - Backbone architecture
    - Ensemble strategy
    - Calibration method
    - Preprocessing variations
    
    Example:
        >>> ablation = AblationStudy(detector, val_dataset, output_dir="results")
        >>> results = ablation.run_full_study()
        >>> ablation.print_summary()
    """

    def __init__(
        self,
        detector,
        val_dataset,
        output_dir: Optional[Union[Path, str]] = None,
        verbose: bool = True,
    ) -> None:
        self.detector = detector
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir) if output_dir else Path("results/ablation")
        self.verbose = verbose
        
        self.results: Dict[str, Any] = {}

    def run_full_study(self) -> Dict[str, Any]:
        """
        Run complete ablation study.
        
        Returns:
            Dictionary with all ablation results
        """
        logger.info("Starting ablation study")
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "baseline": self._evaluate_baseline(),
            "backbone_ablation": self._ablate_backbones(),
            "ensemble_ablation": self._ablate_ensemble_strategies(),
            "calibration_ablation": self._ablate_calibration(),
            "preprocessing_ablation": self._ablate_preprocessing(),
        }
        
        # Compute component importance
        self.results["component_importance"] = self._compute_importance()
        
        return self.results

    def _evaluate_baseline(self) -> Dict[str, Any]:
        """Evaluate baseline model."""
        logger.info("Evaluating baseline model")
        return self._evaluate_model(self.detector)

    def _evaluate_model(self, detector) -> Dict[str, Any]:
        """Evaluate a model on validation data."""
        labels = []
        preds = []
        probs = []
        
        for img, label in self.val_dataset:
            try:
                result = detector.detect(img)
                labels.append(label)
                probs.append(result["ai_probability"])
                preds.append(1 if result["ai_probability"] > 0.5 else 0)
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
        
        if not labels:
            return {"error": "No samples evaluated"}
        
        return compute_metrics(
            np.array(labels),
            np.array(preds),
            np.array(probs),
        )

    def _ablate_backbones(self) -> Dict[str, Any]:
        """Ablate different backbone architectures."""
        logger.info("Ablating backbone architectures")
        
        # This would require creating new detectors with different backbones
        # For now, return placeholder
        backbones = ["efficientnet_b4", "convnext_base", "vit_base_patch16_224"]
        
        results = {}
        for backbone in backbones:
            try:
                from imagetrust.detection import AIDetector
                temp_detector = AIDetector(model=backbone, device=self.detector.device)
                results[backbone] = self._evaluate_model(temp_detector)
            except Exception as e:
                logger.warning(f"Failed to evaluate {backbone}: {e}")
                results[backbone] = {"error": str(e)}
        
        return results

    def _ablate_ensemble_strategies(self) -> Dict[str, Any]:
        """Ablate ensemble strategies."""
        logger.info("Ablating ensemble strategies")
        
        strategies = ["average", "weighted", "voting", "max"]
        
        results = {}
        for strategy in strategies:
            # This would require modifying ensemble strategy
            # For now, evaluate current model
            results[strategy] = {"note": "Requires ensemble reconfiguration"}
        
        return results

    def _ablate_calibration(self) -> Dict[str, Any]:
        """Ablate calibration methods."""
        logger.info("Ablating calibration methods")
        
        results = {
            "with_calibration": self._evaluate_model(self.detector),
            "without_calibration": self._evaluate_without_calibration(),
        }
        
        return results

    def _evaluate_without_calibration(self) -> Dict[str, Any]:
        """Evaluate without calibration."""
        labels = []
        preds = []
        probs = []
        
        for img, label in self.val_dataset:
            try:
                result = self.detector.detect(img, use_calibration=False)
                labels.append(label)
                probs.append(result["ai_probability"])
                preds.append(1 if result["ai_probability"] > 0.5 else 0)
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
        
        if not labels:
            return {"error": "No samples evaluated"}
        
        return compute_metrics(
            np.array(labels),
            np.array(preds),
            np.array(probs),
        )

    def _ablate_preprocessing(self) -> Dict[str, Any]:
        """Ablate preprocessing configurations."""
        logger.info("Ablating preprocessing")
        
        # Different input sizes
        input_sizes = [224, 384, 512]
        
        results = {}
        for size in input_sizes:
            results[f"size_{size}"] = {"note": "Requires preprocessor reconfiguration"}
        
        return results

    def _compute_importance(self) -> Dict[str, float]:
        """
        Compute relative importance of each component.
        
        Uses performance drop when component is removed/modified.
        """
        importance = {}
        
        baseline_f1 = self.results.get("baseline", {}).get("f1_score", 0)
        
        # Calibration importance
        cal_results = self.results.get("calibration_ablation", {})
        with_cal = cal_results.get("with_calibration", {}).get("f1_score", 0)
        without_cal = cal_results.get("without_calibration", {}).get("f1_score", 0)
        importance["calibration"] = max(0, with_cal - without_cal)
        
        # Backbone variance (use std of backbone performances)
        backbone_results = self.results.get("backbone_ablation", {})
        backbone_f1s = [
            r.get("f1_score", 0) 
            for r in backbone_results.values() 
            if isinstance(r, dict) and "f1_score" in r
        ]
        if backbone_f1s:
            importance["backbone_selection"] = np.std(backbone_f1s)
        
        return importance

    def save_results(
        self,
        results: Optional[Dict[str, Any]] = None,
        filename: str = "ablation_results.json",
    ) -> Path:
        """Save results to file."""
        results = results or self.results
        
        ensure_dir(self.output_dir)
        output_path = self.output_dir / filename
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Ablation results saved to {output_path}")
        return output_path

    def print_summary(self) -> None:
        """Print ablation study summary."""
        print("\n" + "=" * 60)
        print("ABLATION STUDY SUMMARY (MLE-STAR Inspired)")
        print("=" * 60)
        
        # Baseline
        baseline = self.results.get("baseline", {})
        print(f"\nBaseline Performance")
        print("-" * 40)
        print(f"  Accuracy:  {baseline.get('accuracy', 0):.2%}")
        print(f"  F1 Score:  {baseline.get('f1_score', 0):.2%}")
        print(f"  ROC-AUC:   {baseline.get('roc_auc', 0):.3f}")
        
        # Component importance
        importance = self.results.get("component_importance", {})
        if importance:
            print(f"\nComponent Importance")
            print("-" * 40)
            for component, score in sorted(importance.items(), key=lambda x: -x[1]):
                print(f"  {component}: {score:.4f}")
        
        # Calibration impact
        cal = self.results.get("calibration_ablation", {})
        if cal:
            print(f"\nCalibration Impact")
            print("-" * 40)
            with_cal = cal.get("with_calibration", {}).get("f1_score", 0)
            without_cal = cal.get("without_calibration", {}).get("f1_score", 0)
            print(f"  With calibration:    F1={with_cal:.2%}")
            print(f"  Without calibration: F1={without_cal:.2%}")
            print(f"  Improvement:         {(with_cal - without_cal):.2%}")
        
        print("=" * 60 + "\n")
