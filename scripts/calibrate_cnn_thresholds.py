#!/usr/bin/env python3
"""
REAL Threshold Calibration for Trained CNN Models
International Publication Level - IEEE WIFS / ACM IH&MMSec

Runs actual model inference on validation data to find:
1. Optimal threshold for F1/Balanced Accuracy/Youden's J
2. UNCERTAIN region (abstain) with precision/recall guarantees
3. Temperature scaling for calibrated probabilities
4. Bootstrap confidence intervals for thresholds
5. Per-threshold performance sweep

Usage:
    python scripts/calibrate_cnn_thresholds.py
    python scripts/calibrate_cnn_thresholds.py --max-samples 5000
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# MODEL ARCHITECTURE (exact match from training)
# ============================================================================

class DeepfakeDetector(nn.Module):
    """Exact architecture from train_kaggle_deepfake.py."""

    def __init__(self, backbone_type="resnet50"):
        super().__init__()
        self.backbone_type = backbone_type

        if backbone_type == "resnet50":
            self.backbone = models.resnet50(weights=None)
            self.backbone.fc = nn.Identity()
            self.num_features = 2048
        elif backbone_type == "efficientnet_v2_m":
            self.backbone = models.efficientnet_v2_m(weights=None)
            self.backbone.classifier = nn.Identity()
            self.num_features = 1280
        elif backbone_type == "convnext_base":
            self.backbone = models.convnext_base(weights=None)
            self.backbone.classifier = nn.Identity()
            self.num_features = 1024

        nf = self.num_features
        self.attention = nn.Sequential(
            nn.Linear(nf, nf // 4), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(nf // 4, nf), nn.Sigmoid()
        )
        self.se = nn.Sequential(
            nn.Linear(nf, nf // 16), nn.ReLU(inplace=True),
            nn.Linear(nf // 16, nf), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(nf), nn.Dropout(0.5),
            nn.Linear(nf, 512), nn.GELU(),
            nn.LayerNorm(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, 2)
        )
        self.dropout_layers = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        elif features.dim() == 3:
            features = features.mean(dim=1)
        features = features * self.attention(features)
        features = features * self.se(features)
        return self.classifier(features)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_validation_data(data_dir: Path, max_samples: int = 5000) -> Tuple[list, np.ndarray]:
    """Load validation data using the same method as academic evaluation."""
    splits_file = data_dir / "splits" / "default_split.json"

    if splits_file.exists():
        print(f"Loading from splits file: {splits_file}")
        with open(splits_file) as f:
            splits = json.load(f)
        val_data = splits.get("val", [])
        if not val_data:
            val_data = splits.get("test", [])
        print(f"  Found {len(val_data)} validation samples from splits")
    else:
        # Fallback to train directory
        print("No splits file found, using train directory...")
        val_data = []
        real_dir = data_dir / "train" / "Real"
        fake_dir = data_dir / "train" / "Fake"

        if real_dir.exists():
            real_images = sorted(real_dir.glob("*.jpg"))[:max_samples // 2]
            for img in real_images:
                val_data.append({"path": str(img), "label": 0})
            print(f"  Real: {len(real_images)} images")

        if fake_dir.exists():
            fake_images = sorted(fake_dir.glob("*.jpg"))[:max_samples // 2]
            for img in fake_images:
                val_data.append({"path": str(img), "label": 1})
            print(f"  Fake: {len(fake_images)} images")

    # Sample if too large
    np.random.seed(42)
    if len(val_data) > max_samples:
        indices = np.random.choice(len(val_data), size=max_samples, replace=False)
        val_data = [val_data[i] for i in indices]

    labels = np.array([item["label"] for item in val_data])
    n_fake = int(np.sum(labels == 1))
    n_real = int(np.sum(labels == 0))
    print(f"  Total: {len(val_data)} (Real: {n_real}, Fake: {n_fake})")

    return val_data, labels


class ValidationDataset(torch.utils.data.Dataset):
    """Dataset for batched inference on validation data."""

    def __init__(self, val_data: list, transform):
        self.val_data = val_data
        self.transform = transform

    def __len__(self):
        return len(self.val_data)

    def __getitem__(self, idx):
        item = self.val_data[idx]
        try:
            img = Image.open(item["path"]).convert("RGB")
            x = self.transform(img)
            return x, idx, True
        except Exception:
            # Return a dummy tensor for failed images
            return torch.zeros(3, 224, 224), idx, False


def get_predictions(
    model: nn.Module, val_data: list, device: str, batch_size: int = 64
) -> np.ndarray:
    """Get model predictions with batched GPU inference. Returns P(AI/Fake)."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ValidationDataset(val_data, transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device != "cpu"),
    )

    probs = np.full(len(val_data), 0.5)  # default 0.5 for failed images
    model.eval()

    with torch.no_grad():
        for batch_x, batch_idx, batch_valid in tqdm(loader, desc="Predicting"):
            batch_x = batch_x.to(device, non_blocking=True)
            out = model(batch_x)
            batch_probs = torch.softmax(out, dim=1)[:, 0].cpu().numpy()

            for i, (idx, valid) in enumerate(zip(batch_idx, batch_valid)):
                if valid:
                    probs[idx.item()] = batch_probs[i]

    return probs


# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

def sweep_thresholds(
    y_true: np.ndarray, probs: np.ndarray, n_thresholds: int = 200
) -> Dict[str, Any]:
    """
    Sweep all thresholds and find optimal ones.
    Returns detailed per-threshold metrics.
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, matthews_corrcoef,
        roc_auc_score, roc_curve
    )

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    sweep_data = []

    for thresh in thresholds:
        y_pred = (probs >= thresh).astype(int)
        row = {
            "threshold": float(thresh),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        }
        sweep_data.append(row)

    # Find optimal thresholds
    best_f1_idx = max(range(len(sweep_data)), key=lambda i: sweep_data[i]["f1"])
    best_ba_idx = max(range(len(sweep_data)), key=lambda i: sweep_data[i]["balanced_accuracy"])
    best_mcc_idx = max(range(len(sweep_data)), key=lambda i: sweep_data[i]["mcc"])

    # Youden's J
    fpr, tpr, roc_thresholds = roc_curve(y_true, probs)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    youden_threshold = float(roc_thresholds[best_j_idx])

    # AUC (threshold-independent)
    auc_val = float(roc_auc_score(y_true, probs))

    optimal_thresholds = {
        "f1_optimal": {
            "threshold": sweep_data[best_f1_idx]["threshold"],
            "metrics": sweep_data[best_f1_idx],
        },
        "balanced_accuracy_optimal": {
            "threshold": sweep_data[best_ba_idx]["threshold"],
            "metrics": sweep_data[best_ba_idx],
        },
        "mcc_optimal": {
            "threshold": sweep_data[best_mcc_idx]["threshold"],
            "metrics": sweep_data[best_mcc_idx],
        },
        "youden_optimal": {
            "threshold": youden_threshold,
        },
        "default_0.5": {
            "threshold": 0.5,
            "metrics": sweep_data[min(range(len(sweep_data)),
                                      key=lambda i: abs(sweep_data[i]["threshold"] - 0.5))],
        },
    }

    return {
        "sweep": sweep_data,
        "optimal": optimal_thresholds,
        "auc": auc_val,
    }


def find_uncertain_region(
    y_true: np.ndarray,
    probs: np.ndarray,
    target_accuracy: float = 0.90,
) -> Dict[str, Any]:
    """
    Find UNCERTAIN region thresholds.

    Strategy: Find thresholds such that predictions OUTSIDE the uncertain
    region have >= target_accuracy.

    Sweep symmetric and asymmetric configurations.
    """
    results = []

    # Try different low/high threshold combinations
    for low in np.arange(0.10, 0.60, 0.02):
        for high in np.arange(max(low + 0.05, 0.40), 0.95, 0.02):
            # Predictions: below low = REAL (0), above high = AI (1)
            confident_mask = (probs < low) | (probs >= high)
            n_confident = np.sum(confident_mask)

            if n_confident < 10:
                continue

            confident_preds = np.where(probs[confident_mask] >= high, 1, 0)
            confident_acc = np.mean(confident_preds == y_true[confident_mask])
            coverage = n_confident / len(probs)

            # Precision/Recall for confident predictions
            tp = np.sum((confident_preds == 1) & (y_true[confident_mask] == 1))
            tn = np.sum((confident_preds == 0) & (y_true[confident_mask] == 0))
            fp = np.sum((confident_preds == 1) & (y_true[confident_mask] == 0))
            fn = np.sum((confident_preds == 0) & (y_true[confident_mask] == 1))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0

            results.append({
                "low": float(low),
                "high": float(high),
                "coverage": float(coverage),
                "accuracy": float(confident_acc),
                "precision": float(prec),
                "recall": float(rec),
                "n_confident": int(n_confident),
                "n_uncertain": int(len(probs) - n_confident),
            })

    if not results:
        return {"error": "Could not find valid uncertain region"}

    # Find best configuration that meets accuracy target with max coverage
    valid = [r for r in results if r["accuracy"] >= target_accuracy]

    if valid:
        best = max(valid, key=lambda r: r["coverage"])
        method = f"target_{target_accuracy:.0%}_accuracy"
    else:
        # If can't meet target, find the one closest to target with decent coverage
        best = min(results, key=lambda r: abs(r["accuracy"] - target_accuracy))
        method = "closest_to_target"

    # Also find configurations for different targets
    configs = {}
    for target in [0.85, 0.90, 0.95]:
        valid_t = [r for r in results if r["accuracy"] >= target]
        if valid_t:
            configs[f"target_{int(target*100)}pct"] = max(valid_t, key=lambda r: r["coverage"])

    return {
        "recommended": best,
        "method": method,
        "target_accuracy": target_accuracy,
        "configs_by_target": configs,
        "n_configs_evaluated": len(results),
    }


def temperature_scaling(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """Find optimal temperature for calibration."""
    from scipy.optimize import minimize_scalar
    from scipy.special import expit

    # Convert to logits
    probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    logits = np.log(probs_clipped / (1 - probs_clipped))

    def nll(T):
        scaled = expit(logits / T)
        scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    optimal_T = result.x
    calibrated = expit(logits / optimal_T)

    return optimal_T, calibrated


def compute_ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> Tuple[float, dict]:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    reliability = {"bin_centers": [], "accuracies": [], "confidences": [], "counts": []}

    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (probs > lower) & (probs <= upper)
        if np.sum(mask) > 0:
            acc = np.mean(labels[mask])
            conf = np.mean(probs[mask])
            prop = np.mean(mask)
            ece += prop * abs(acc - conf)

            reliability["bin_centers"].append(float((lower + upper) / 2))
            reliability["accuracies"].append(float(acc))
            reliability["confidences"].append(float(conf))
            reliability["counts"].append(int(np.sum(mask)))

    return float(ece), reliability


def bootstrap_threshold_ci(
    y_true: np.ndarray, probs: np.ndarray,
    n_bootstrap: int = 1000, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Bootstrap CI for optimal F1 threshold."""
    from sklearn.metrics import f1_score

    np.random.seed(42)
    boot_thresholds = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        boot_labels = y_true[idx]
        boot_probs = probs[idx]

        best_t, best_f1 = 0.5, 0
        for t in np.arange(0.1, 0.9, 0.02):
            preds = (boot_probs >= t).astype(int)
            f1 = f1_score(boot_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        boot_thresholds.append(best_t)

    boot_thresholds = np.array(boot_thresholds)
    alpha = 1 - confidence
    lower = np.percentile(boot_thresholds, 100 * alpha / 2)
    upper = np.percentile(boot_thresholds, 100 * (1 - alpha / 2))
    mean_t = np.mean(boot_thresholds)

    return float(mean_t), float(lower), float(upper)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Calibrate CNN thresholds (REAL)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/threshold_calibration"))
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap iterations")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REAL CNN THRESHOLD CALIBRATION")
    print("International Publication Level")
    print("=" * 70)
    print(f"Data dir: {args.data_dir}")
    print(f"Device: {args.device}")
    print(f"Max samples: {args.max_samples}")
    print(f"Bootstrap iterations: {args.bootstrap}")
    print()

    # Load validation data
    val_data, y_true = load_validation_data(args.data_dir, args.max_samples)

    if len(val_data) == 0:
        print("ERROR: No validation data found!")
        sys.exit(1)

    # Model configs
    model_configs = [
        ("ResNet-50", "outputs/training_resnet50/best_model.pth", "resnet50"),
        ("EfficientNetV2-M", "outputs/training_efficientnet/best_model.pth", "efficientnet_v2_m"),
        ("ConvNeXt-Base", "outputs/training_convnext/best_model.pth", "convnext_base"),
    ]

    results = {}
    all_probs = {}

    for model_name, model_path, backbone in model_configs:
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"\nSkipping {model_name}: {model_path} not found")
            continue

        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")

        # Load model
        model = DeepfakeDetector(backbone_type=backbone)
        cp = torch.load(model_path, map_location=args.device, weights_only=False)
        model.load_state_dict(cp["model_state_dict"])
        model.to(args.device)
        model.eval()

        # Get predictions
        probs = get_predictions(model, val_data, args.device)
        all_probs[model_name] = probs

        print(f"\nProbability distribution:")
        print(f"  Range: [{probs.min():.4f}, {probs.max():.4f}]")
        print(f"  Mean: {probs.mean():.4f}, Std: {probs.std():.4f}")
        print(f"  Median: {np.median(probs):.4f}")
        print(f"  P25: {np.percentile(probs, 25):.4f}, P75: {np.percentile(probs, 75):.4f}")

        # Histogram of predictions
        for p_range in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
            count = np.sum((probs >= p_range[0]) & (probs < p_range[1]))
            bar = "#" * (count * 50 // len(probs))
            print(f"  [{p_range[0]:.1f}-{p_range[1]:.1f}]: {count:5d} ({count/len(probs)*100:5.1f}%) {bar}")

        # Threshold sweep
        print("\nSweeping thresholds...")
        sweep_result = sweep_thresholds(y_true, probs)

        print("\nOptimal thresholds found:")
        for method, data in sweep_result["optimal"].items():
            t = data["threshold"]
            m = data.get("metrics", {})
            f1 = m.get("f1", 0) * 100
            acc = m.get("accuracy", 0) * 100
            ba = m.get("balanced_accuracy", 0) * 100
            prec = m.get("precision", 0) * 100
            rec = m.get("recall", 0) * 100
            print(f"  {method:30s}: t={t:.4f}  F1={f1:.1f}%  Acc={acc:.1f}%  BA={ba:.1f}%  P={prec:.1f}%  R={rec:.1f}%")

        print(f"  AUC (threshold-independent): {sweep_result['auc']*100:.2f}%")

        # UNCERTAIN region
        print("\nFinding UNCERTAIN region...")
        uncertain = find_uncertain_region(y_true, probs, target_accuracy=0.90)

        if "recommended" in uncertain:
            rec = uncertain["recommended"]
            print(f"\n  Recommended UNCERTAIN region:")
            print(f"    Low threshold:  {rec['low']:.2f}  (below = REAL confident)")
            print(f"    High threshold: {rec['high']:.2f}  (above = AI confident)")
            print(f"    Coverage:       {rec['coverage']*100:.1f}%  (confident predictions)")
            print(f"    Accuracy:       {rec['accuracy']*100:.1f}%  (on confident predictions)")
            print(f"    Uncertain:      {rec['n_uncertain']} samples ({(1-rec['coverage'])*100:.1f}%)")

            if "configs_by_target" in uncertain:
                print(f"\n  Configurations by target accuracy:")
                for name, cfg in uncertain["configs_by_target"].items():
                    print(f"    {name}: low={cfg['low']:.2f}, high={cfg['high']:.2f}, "
                          f"coverage={cfg['coverage']*100:.1f}%, acc={cfg['accuracy']*100:.1f}%")

        # Temperature scaling
        print("\nTemperature Scaling...")
        temperature, calibrated_probs = temperature_scaling(probs, y_true)
        ece_before, rel_before = compute_ece(y_true, probs)
        ece_after, rel_after = compute_ece(y_true, calibrated_probs)

        print(f"  Temperature: {temperature:.4f}")
        print(f"  ECE: {ece_before*100:.2f}% -> {ece_after*100:.2f}%")

        # Bootstrap CI
        print(f"\nBootstrap CI ({args.bootstrap} iterations)...")
        boot_mean, boot_lower, boot_upper = bootstrap_threshold_ci(
            y_true, probs, n_bootstrap=args.bootstrap
        )
        print(f"  F1-optimal threshold: {boot_mean:.4f} [{boot_lower:.4f}, {boot_upper:.4f}]")

        # Store results
        f1_thresh = sweep_result["optimal"]["f1_optimal"]["threshold"]

        results[model_name] = {
            "recommended_threshold": f1_thresh,
            "threshold_sweep": {
                k: v for k, v in sweep_result["optimal"].items()
            },
            "auc": sweep_result["auc"],
            "uncertain_region": uncertain,
            "temperature_scaling": {
                "temperature": temperature,
                "ece_before": ece_before,
                "ece_after": ece_after,
                "reliability_before": rel_before,
                "reliability_after": rel_after,
            },
            "bootstrap_ci": {
                "mean": boot_mean,
                "lower_95": boot_lower,
                "upper_95": boot_upper,
            },
            "probability_distribution": {
                "min": float(probs.min()),
                "max": float(probs.max()),
                "mean": float(probs.mean()),
                "std": float(probs.std()),
                "median": float(np.median(probs)),
                "p25": float(np.percentile(probs, 25)),
                "p75": float(np.percentile(probs, 75)),
            },
        }

        # Save per-threshold sweep data
        sweep_file = args.output / f"sweep_{model_name.replace(' ', '_').replace('-', '')}.json"
        with open(sweep_file, "w") as f:
            json.dump(sweep_result["sweep"], f, indent=2)

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ========================================================================
    # ENSEMBLE
    # ========================================================================
    if len(all_probs) >= 2:
        print(f"\n{'='*70}")
        print("ENSEMBLE CALIBRATION")
        print(f"{'='*70}")

        for strategy_name, strategy_fn in [
            ("average", lambda p: np.mean(p, axis=0)),
            ("min", lambda p: np.min(p, axis=0)),
        ]:
            ens_probs = strategy_fn(np.stack(list(all_probs.values())))

            sweep = sweep_thresholds(y_true, ens_probs)
            uncertain = find_uncertain_region(y_true, ens_probs)
            temperature, cal_probs = temperature_scaling(ens_probs, y_true)
            ece_b, _ = compute_ece(y_true, ens_probs)
            ece_a, _ = compute_ece(y_true, cal_probs)

            f1_t = sweep["optimal"]["f1_optimal"]["threshold"]
            f1_m = sweep["optimal"]["f1_optimal"]["metrics"]

            print(f"\n  Ensemble ({strategy_name}):")
            print(f"    F1-optimal threshold: {f1_t:.4f}")
            print(f"    F1: {f1_m['f1']*100:.1f}%, Acc: {f1_m['accuracy']*100:.1f}%")
            print(f"    AUC: {sweep['auc']*100:.2f}%")
            print(f"    Temperature: {temperature:.4f}, ECE: {ece_b*100:.2f}% -> {ece_a*100:.2f}%")

            if "recommended" in uncertain:
                rec = uncertain["recommended"]
                print(f"    UNCERTAIN: [{rec['low']:.2f}, {rec['high']:.2f}], "
                      f"coverage={rec['coverage']*100:.1f}%, acc={rec['accuracy']*100:.1f}%")

            results[f"Ensemble ({strategy_name})"] = {
                "recommended_threshold": f1_t,
                "threshold_sweep": sweep["optimal"],
                "auc": sweep["auc"],
                "uncertain_region": uncertain,
                "temperature_scaling": {
                    "temperature": temperature,
                    "ece_before": ece_b,
                    "ece_after": ece_a,
                },
            }

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    output_file = args.output / "threshold_calibration_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "date": datetime.now().isoformat(),
                "data_dir": str(args.data_dir),
                "n_samples": len(y_true),
                "n_fake": int(np.sum(y_true == 1)),
                "n_real": int(np.sum(y_true == 0)),
                "seed": 42,
                "bootstrap_iterations": args.bootstrap,
            },
            "models": results,
        }, f, indent=2)

    print(f"\n{'='*70}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_file}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Threshold':<12} {'95% CI':<20} {'F1':<8} {'AUC':<8} {'Temp':<8}")
    print("-" * 85)

    for name, data in results.items():
        t = data["recommended_threshold"]
        ci = data.get("bootstrap_ci", {})
        ci_str = f"[{ci.get('lower_95', 0):.3f}, {ci.get('upper_95', 0):.3f}]" if ci else "N/A"
        f1_m = data["threshold_sweep"]["f1_optimal"].get("metrics", {})
        f1 = f1_m.get("f1", 0) * 100
        auc_val = data.get("auc", 0) * 100
        temp = data.get("temperature_scaling", {}).get("temperature", 0)
        print(f"{name:<25} {t:<12.4f} {ci_str:<20} {f1:<8.1f} {auc_val:<8.1f} {temp:<8.3f}")

    print(f"\nUse the recommended threshold for binary classification.")
    print(f"Use UNCERTAIN region thresholds for 3-way classification (Real/AI/Uncertain).")


if __name__ == "__main__":
    main()
