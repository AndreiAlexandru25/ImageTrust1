#!/usr/bin/env python3
"""
Example 4: Probability Calibration

Demonstrates how calibration improves reliability of confidence estimates.

Usage:
    python examples/04_calibration_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def demonstrate_calibration():
    """Demonstrate the effect of probability calibration."""
    import numpy as np

    print("\n" + "=" * 70)
    print("ImageTrust - Probability Calibration Demo")
    print("=" * 70)

    # Simulate uncalibrated predictions (overconfident)
    np.random.seed(42)
    n_samples = 1000

    # True labels
    labels = np.random.randint(0, 2, n_samples)

    # Uncalibrated predictions (model is overconfident)
    # AI images predicted with high confidence
    # Real images also have predictions pushed toward extremes
    raw_probs = np.zeros(n_samples)
    for i in range(n_samples):
        if labels[i] == 1:  # AI
            raw_probs[i] = np.clip(0.85 + np.random.normal(0, 0.15), 0.5, 0.99)
        else:  # Real
            raw_probs[i] = np.clip(0.15 + np.random.normal(0, 0.15), 0.01, 0.5)

    # Add some noise and push toward extremes (overconfidence)
    raw_probs = np.clip(raw_probs ** 0.5 if np.random.rand() > 0.5 else raw_probs ** 1.5, 0.01, 0.99)

    print("\n[1] Before Calibration (Raw Model Output)")
    print("-" * 50)

    # Calculate ECE before calibration
    ece_before = calculate_ece(labels, raw_probs)
    accuracy = np.mean((raw_probs > 0.5) == labels)

    print(f"Accuracy: {accuracy:.1%}")
    print(f"Expected Calibration Error (ECE): {ece_before:.4f}")
    print_calibration_bins(labels, raw_probs)

    # Apply temperature scaling (simulated)
    print("\n[2] Applying Temperature Scaling")
    print("-" * 50)

    # Find optimal temperature
    best_temp = find_optimal_temperature(labels, raw_probs)
    print(f"Optimal temperature: {best_temp:.3f}")

    # Apply calibration
    calibrated_probs = apply_temperature_scaling(raw_probs, best_temp)

    print("\n[3] After Calibration")
    print("-" * 50)

    ece_after = calculate_ece(labels, calibrated_probs)
    accuracy_after = np.mean((calibrated_probs > 0.5) == labels)

    print(f"Accuracy: {accuracy_after:.1%}")
    print(f"Expected Calibration Error (ECE): {ece_after:.4f}")
    print_calibration_bins(labels, calibrated_probs)

    # Summary
    print("\n" + "=" * 70)
    print("Calibration Summary")
    print("=" * 70)
    print(f"{'Metric':<25} {'Before':>15} {'After':>15} {'Change':>15}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {accuracy:>14.1%} {accuracy_after:>14.1%} {'+0.0%':>15}")
    print(f"{'ECE':<25} {ece_before:>15.4f} {ece_after:>15.4f} {ece_after-ece_before:>+15.4f}")
    print(f"{'ECE Improvement':<25} {'-':>15} {'-':>15} {(ece_before-ece_after)/ece_before*100:>14.1f}%")
    print("-" * 70)

    print("\nKey Insight:")
    print("  - Calibration does NOT change accuracy (same predictions)")
    print("  - Calibration DOES improve reliability of confidence estimates")
    print("  - Lower ECE means model confidence matches actual accuracy")


def calculate_ece(labels: 'np.ndarray', probs: 'np.ndarray', n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error."""
    import numpy as np

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accuracy = labels[mask].mean()
            bin_confidence = probs[mask].mean()
            bin_weight = mask.sum() / len(probs)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


def print_calibration_bins(labels: 'np.ndarray', probs: 'np.ndarray', n_bins: int = 5):
    """Print calibration breakdown by confidence bin."""
    import numpy as np

    print("\nCalibration by confidence bin:")
    print(f"  {'Bin':>12} {'Count':>8} {'Confidence':>12} {'Accuracy':>12} {'Gap':>8}")
    print("  " + "-" * 56)

    bin_edges = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        count = mask.sum()

        if count > 0:
            conf = probs[mask].mean()
            acc = labels[mask].mean()
            gap = abs(conf - acc)

            bin_label = f"[{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f})"
            print(f"  {bin_label:>12} {count:>8} {conf:>11.1%} {acc:>11.1%} {gap:>7.1%}")
        else:
            bin_label = f"[{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f})"
            print(f"  {bin_label:>12} {0:>8} {'N/A':>12} {'N/A':>12} {'N/A':>8}")


def find_optimal_temperature(labels: 'np.ndarray', probs: 'np.ndarray') -> float:
    """Find optimal temperature for scaling."""
    import numpy as np

    best_temp = 1.0
    best_ece = calculate_ece(labels, probs)

    for temp in np.arange(0.5, 3.0, 0.1):
        cal_probs = apply_temperature_scaling(probs, temp)
        ece = calculate_ece(labels, cal_probs)
        if ece < best_ece:
            best_ece = ece
            best_temp = temp

    return best_temp


def apply_temperature_scaling(probs: 'np.ndarray', temperature: float) -> 'np.ndarray':
    """Apply temperature scaling to probabilities."""
    import numpy as np

    # Convert to logits
    eps = 1e-7
    probs = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))

    # Scale by temperature
    scaled_logits = logits / temperature

    # Convert back to probabilities
    return 1 / (1 + np.exp(-scaled_logits))


def main():
    try:
        demonstrate_calibration()
    except ImportError as e:
        print(f"Note: Some dependencies missing ({e})")
        print("This example requires numpy. Install with: pip install numpy")


if __name__ == "__main__":
    main()
