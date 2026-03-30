#!/usr/bin/env python3
"""
Regenerate paper figures with professor's fixes:
- Serif font (Times) matching the paper
- No background colors on legends (frameon=False)
- Clean white backgrounds, no colored grid
- Heatmap: simplified axis labels, matching font
"""

import json
import sys
from pathlib import Path

import numpy as np

PHASE3_DIR = Path("outputs/phase3")
OUTPUT_DIR = Path("publicatie_overleaf/figures")


def setup_matplotlib():
    """Configure matplotlib for publication - serif font, clean style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 14,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "legend.frameon": False,
        "legend.fancybox": False,
    })
    return plt


# ── Colors ──
COLORS = {
    "logreg": "#7570b3",
    "single_resnet50": "#e7298a",
    "single_efficientnet_b0": "#66a61e",
    "single_vit_b_16": "#d95f02",
    "xgboost_meta": "#1b9e77",
    "mlp_meta": "#e6ab02",
}
DISPLAY_NAMES = {
    "logreg": "B1: LogReg (ResNet-50 emb.)",
    "single_resnet50": "B2: XGB (ResNet-50 emb.)",
    "single_efficientnet_b0": "B2: XGB (EffNet-B0 emb.)",
    "single_vit_b_16": "B3: XGB (ViT-B/16 emb.)",
    "xgboost_meta": "Ours: XGB (3-backbone fusion)",
    "mlp_meta": "Ours: MLP (3-backbone fusion)",
}
LINESTYLES = {
    "logreg": ":",
    "single_resnet50": "--",
    "single_efficientnet_b0": "-.",
    "single_vit_b_16": "--",
    "xgboost_meta": "-",
    "mlp_meta": "-",
}
LINEWIDTHS = {
    "logreg": 1.5,
    "single_resnet50": 1.5,
    "single_efficientnet_b0": 1.5,
    "single_vit_b_16": 1.5,
    "xgboost_meta": 2.0,
    "mlp_meta": 2.5,
}


def regenerate_fig1_roc():
    """Fig 2 in paper: ROC curves with bootstrap CI."""
    from sklearn.metrics import roc_curve, auc

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5, 4))

    method_order = [
        "single_resnet50", "single_efficientnet_b0",
        "single_vit_b_16", "logreg",
        "xgboost_meta", "mlp_meta",
    ]

    baselines_dir = PHASE3_DIR / "baselines"
    train_dir = PHASE3_DIR / "train_predictions"

    for method in method_order:
        # Load predictions
        pred_file = baselines_dir / f"{method}_predictions.npz"
        if not pred_file.exists():
            pred_file = train_dir / f"{method}_train_predictions.npz"
        if not pred_file.exists():
            print(f"  Skipping {method}: no predictions file")
            continue

        d = np.load(pred_file, allow_pickle=True)
        # Handle both naming conventions
        if "y_proba_test" in d:
            y_proba = d["y_proba_test"]
            y_true = d["y_test"]
        elif "test_proba" in d:
            y_proba = d["test_proba"]
            y_true = d["test_labels"]
        else:
            print(f"  Skipping {method}: unknown keys {list(d.keys())}")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        color = COLORS.get(method, "gray")
        ls = LINESTYLES.get(method, "-")
        lw = LINEWIDTHS.get(method, 1.5)
        display = DISPLAY_NAMES.get(method, method)

        # Bootstrap CI
        rng = np.random.RandomState(42)
        n = len(y_true)
        mean_fpr = np.linspace(0, 1, 100)
        tpr_list = []
        for _ in range(200):
            idx = rng.choice(n, n, replace=True)
            if len(np.unique(y_true[idx])) < 2:
                continue
            fpr_b, tpr_b, _ = roc_curve(y_true[idx], y_proba[idx])
            tpr_list.append(np.interp(mean_fpr, fpr_b, tpr_b))

        if tpr_list:
            tpr_stack = np.array(tpr_list)
            ax.fill_between(
                mean_fpr,
                np.percentile(tpr_stack, 2.5, axis=0),
                np.percentile(tpr_stack, 97.5, axis=0),
                alpha=0.08, color=color,
            )

        ax.plot(fpr, tpr, label=f"{display} (AUC={roc_auc:.3f})",
                color=color, linestyle=ls, linewidth=lw)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves with 95% Bootstrap CI")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUTPUT_DIR / "fig1_roc_curves.pdf"
    plt.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"  Saved: {out}")


def regenerate_fig2_reliability():
    """Fig 3 in paper: Reliability diagrams 2x2."""
    plt = setup_matplotlib()

    with open(PHASE3_DIR / "calibration" / "all_calibration_methods.json") as f:
        cal_data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(7, 6))

    methods = [
        ("xgboost_meta", "XGBoost Meta-Classifier"),
        ("mlp_meta", "MLP Meta-Classifier"),
    ]

    for row, (method_key, method_label) in enumerate(methods):
        md = cal_data.get(method_key, {})

        # Column 0: Uncalibrated
        rd_raw = md.get("reliability_uncalibrated", {})
        ax = axes[row, 0]
        if rd_raw and rd_raw.get("bin_centers"):
            ax.bar(
                rd_raw["bin_centers"], rd_raw["bin_accuracies"],
                width=1.0 / rd_raw["n_bins"], alpha=0.6,
                color="steelblue", edgecolor="navy", linewidth=0.5,
            )
            ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect")
            ece_val = md.get("uncalibrated", {}).get("ece", 0)
            ax.set_title(f"{method_label}\nUncalibrated (ECE={ece_val:.4f})")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Column 1: Best calibrated (isotonic)
        best_cal = None
        best_ece = 999
        for cal_type in ["temperature", "platt", "isotonic"]:
            rd_cal = md.get(f"reliability_{cal_type}", {})
            ece_c = md.get(cal_type, {}).get("ece", 999)
            if rd_cal and rd_cal.get("bin_centers") and ece_c < best_ece:
                best_cal = (cal_type, rd_cal, ece_c)
                best_ece = ece_c

        ax = axes[row, 1]
        if best_cal:
            cal_type, rd_cal, ece_c = best_cal
            ax.bar(
                rd_cal["bin_centers"], rd_cal["bin_accuracies"],
                width=1.0 / rd_cal["n_bins"], alpha=0.6,
                color="forestgreen", edgecolor="darkgreen", linewidth=0.5,
            )
            ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect")
            ax.set_title(f"{method_label}\n{cal_type.title()} (ECE={ece_c:.4f})")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Reliability Diagrams: Before and After Calibration", fontsize=16)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig2_reliability_diagrams.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"  Saved: {out}")


def regenerate_fig3_heatmap():
    """Fig 4 in paper: Cross-source heatmap."""
    plt = setup_matplotlib()

    with open(PHASE3_DIR / "cross_source" / "per_source_metrics.json") as f:
        cross_source = json.load(f)

    method_order = [
        m for m in [
            "logreg", "single_resnet50", "single_vit_b_16",
            "xgboost_meta", "mlp_meta",
        ] if m in cross_source
    ]

    all_sources = set()
    for m in method_order:
        all_sources.update(cross_source[m].keys())
    source_order = sorted(all_sources)

    # Simplified source labels (no "dataset" names)
    SOURCE_LABELS = {
        "CIFAKE-Real": "Real (CIFAR)",
        "CIFAKE-SD": "SD (CIFAR)",
        "COCO": "Real (COCO)",
        "Deepfake-Fake": "Fake (faces)",
        "Deepfake-Real": "Real (faces)",
        "FFHQ": "Real (FFHQ)",
        "Other": "Other",
        "SFHQ": "Synth. (SFHQ)",
    }

    matrix = np.full((len(method_order), len(source_order)), np.nan)
    for i, m in enumerate(method_order):
        for j, s in enumerate(source_order):
            val = cross_source.get(m, {}).get(s, {})
            if isinstance(val, dict) and "roc_auc" in val:
                matrix[i, j] = val["roc_auc"]
            elif isinstance(val, dict) and "accuracy" in val:
                matrix[i, j] = val["accuracy"]

    ylabels = [DISPLAY_NAMES.get(m, m) for m in method_order]
    xlabels = [SOURCE_LABELS.get(s, s) for s in source_order]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Use imshow for full control over text colors
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(source_order)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(method_order)))
    ax.set_yticklabels(ylabels)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=12)

    # Add text with automatic white/black contrast
    for i in range(len(method_order)):
        for j in range(len(source_order)):
            if not np.isnan(matrix[i, j]):
                # White text on dark cells, black on light
                text_color = "white" if matrix[i, j] > 0.85 else "black"
                ax.text(j, i, f"{matrix[i,j]:.3f}",
                        ha="center", va="center", fontsize=11,
                        color=text_color, fontweight="bold")

    ax.set_title("Cross-Source Performance (AUC)")
    plt.tight_layout()
    out = OUTPUT_DIR / "fig3_cross_source_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    print("Regenerating paper figures with professor fixes...")
    print(f"  Output: {OUTPUT_DIR}\n")

    # Check xgboost/mlp predictions exist
    meta_preds = PHASE3_DIR / "baselines"
    if not meta_preds.exists():
        print("ERROR: outputs/phase3/baselines/ not found")
        sys.exit(1)

    # Check for meta-classifier predictions in baselines dir
    xgb_pred = meta_preds / "xgboost_meta_predictions.npz"
    mlp_pred = meta_preds / "mlp_meta_predictions.npz"
    if not xgb_pred.exists():
        # Try train_predictions dir
        alt = PHASE3_DIR / "train_predictions"
        if (alt / "xgboost_meta_train_predictions.npz").exists():
            print("  Note: using train_predictions for meta-classifiers")

    print("--- Fig 1: ROC Curves ---")
    regenerate_fig1_roc()

    print("\n--- Fig 2: Reliability Diagrams ---")
    regenerate_fig2_reliability()

    print("\n--- Fig 3: Cross-Source Heatmap ---")
    regenerate_fig3_heatmap()

    print("\nDone! Check publicatie_overleaf/figures/")
