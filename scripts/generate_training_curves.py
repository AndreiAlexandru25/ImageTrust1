"""
Generate training curves for Phase 2 meta-classifiers.

Produces:
  fig9_training_curves.png / .pdf  — 2x2 grid with:
    (a) XGBoost Log-Loss (train vs val, 768 rounds)
    (b) XGBoost AUC (train vs val, 768 rounds)
    (c) MLP Loss (train vs val, 36 epochs)
    (d) MLP AUC (val) + Learning Rate schedule
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ROOT / "models" / "phase2"
OUT = ROOT / "outputs" / "phase3" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Style matching existing figures ──────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    # ── Load data ────────────────────────────────────────────────────────
    xgb = load_json(MODELS / "xgboost" / "training_history.json")
    mlp = load_json(MODELS / "mlp" / "training_history.json")

    xgb_rounds = np.arange(1, len(xgb["train"]["logloss"]) + 1)
    mlp_epochs = np.arange(1, len(mlp["train_loss"]) + 1)

    # ── Create 2x2 figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Training Curves for Phase 2 Meta-Classifiers", fontsize=15, y=0.98)

    # ── (a) XGBoost Log-Loss ─────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(xgb_rounds, xgb["train"]["logloss"], color="#2196F3", linewidth=1.2,
            label="Train", alpha=0.85)
    ax.plot(xgb_rounds, xgb["val"]["logloss"], color="#F44336", linewidth=1.2,
            label="Validation", alpha=0.85)

    # Mark best validation
    best_val_idx = int(np.argmin(xgb["val"]["logloss"]))
    best_val = xgb["val"]["logloss"][best_val_idx]
    ax.axvline(best_val_idx + 1, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.annotate(f"Best: {best_val:.4f}\n(round {best_val_idx + 1})",
                xy=(best_val_idx + 1, best_val),
                xytext=(best_val_idx + 80, best_val + 0.05),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("Log-Loss")
    ax.set_title("(a) XGBoost Meta-Classifier — Log-Loss")
    ax.legend(loc="upper right")

    # ── (b) XGBoost AUC ─────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(xgb_rounds, xgb["train"]["auc"], color="#2196F3", linewidth=1.2,
            label="Train", alpha=0.85)
    ax.plot(xgb_rounds, xgb["val"]["auc"], color="#F44336", linewidth=1.2,
            label="Validation", alpha=0.85)

    # Mark best validation AUC
    best_auc_idx = int(np.argmax(xgb["val"]["auc"]))
    best_auc = xgb["val"]["auc"][best_auc_idx]
    ax.axhline(best_auc, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.annotate(f"Best: {best_auc:.4f}\n(round {best_auc_idx + 1})",
                xy=(best_auc_idx + 1, best_auc),
                xytext=(best_auc_idx - 200, best_auc - 0.015),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("(b) XGBoost Meta-Classifier — AUC-ROC")
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # ── (c) MLP Loss ─────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(mlp_epochs, mlp["train_loss"], color="#2196F3", linewidth=1.5,
            marker="o", markersize=3, label="Train", alpha=0.85)
    ax.plot(mlp_epochs, mlp["val_loss"], color="#F44336", linewidth=1.5,
            marker="s", markersize=3, label="Validation", alpha=0.85)

    # Mark best validation loss
    best_mlp_idx = int(np.argmin(mlp["val_loss"]))
    best_mlp_val = mlp["val_loss"][best_mlp_idx]
    ax.axvline(mlp_epochs[best_mlp_idx], color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.annotate(f"Best: {best_mlp_val:.4f}\n(epoch {mlp_epochs[best_mlp_idx]})",
                xy=(mlp_epochs[best_mlp_idx], best_mlp_val),
                xytext=(mlp_epochs[best_mlp_idx] + 5, best_mlp_val + 0.02),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

    # Early stopping region
    if best_mlp_idx < len(mlp_epochs) - 1:
        ax.axvspan(mlp_epochs[best_mlp_idx], mlp_epochs[-1], alpha=0.06, color="red",
                   label="Overfitting region")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("(c) MLP Meta-Classifier — Loss")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ── (d) MLP AUC + Learning Rate ─────────────────────────────────────
    ax = axes[1, 1]
    color_auc = "#4CAF50"
    color_lr = "#FF9800"

    ax.plot(mlp_epochs, mlp["val_auc"], color=color_auc, linewidth=1.5,
            marker="D", markersize=3, label="Validation AUC", alpha=0.85)

    # Best AUC
    best_auc_mlp_idx = int(np.argmax(mlp["val_auc"]))
    best_auc_mlp = mlp["val_auc"][best_auc_mlp_idx]
    ax.axhline(best_auc_mlp, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.annotate(f"Best: {best_auc_mlp:.4f}\n(epoch {mlp_epochs[best_auc_mlp_idx]})",
                xy=(mlp_epochs[best_auc_mlp_idx], best_auc_mlp),
                xytext=(mlp_epochs[best_auc_mlp_idx] + 6, best_auc_mlp - 0.005),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC-ROC", color=color_auc)
    ax.tick_params(axis="y", labelcolor=color_auc)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Secondary axis for learning rate
    ax2 = ax.twinx()
    ax2.plot(mlp_epochs[:len(mlp["lr"])], mlp["lr"], color=color_lr,
             linewidth=1.2, linestyle="--", label="Learning Rate", alpha=0.7)
    ax2.set_ylabel("Learning Rate", color=color_lr)
    ax2.tick_params(axis="y", labelcolor=color_lr)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))

    ax.set_title("(d) MLP Meta-Classifier — AUC-ROC + LR Schedule")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ── Layout and save ──────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ("png", "pdf"):
        path = OUT / f"fig9_training_curves.{ext}"
        fig.savefig(path)
        print(f"Saved: {path}")

    plt.close(fig)

    # ── Print summary stats ──────────────────────────────────────────────
    print("\n=== XGBoost (768 boosting rounds) ===")
    print(f"  Train logloss: {xgb['train']['logloss'][0]:.4f} -> {xgb['train']['logloss'][-1]:.4f}")
    print(f"  Val   logloss: {xgb['val']['logloss'][0]:.4f} -> {xgb['val']['logloss'][-1]:.4f}")
    print(f"  Best val logloss: {best_val:.4f} at round {best_val_idx + 1}")
    print(f"  Train AUC: {xgb['train']['auc'][0]:.4f} -> {xgb['train']['auc'][-1]:.4f}")
    print(f"  Val   AUC: {xgb['val']['auc'][0]:.4f} -> {xgb['val']['auc'][-1]:.4f}")
    print(f"  Best val AUC: {best_auc:.4f} at round {best_auc_idx + 1}")
    print(f"  Train error: {xgb['train']['error'][0]:.4f} -> {xgb['train']['error'][-1]:.4f}")
    print(f"  Val   error: {xgb['val']['error'][0]:.4f} -> {xgb['val']['error'][-1]:.4f}")

    print(f"\n=== MLP (36 epochs, cosine LR schedule) ===")
    print(f"  Train loss: {mlp['train_loss'][0]:.4f} -> {mlp['train_loss'][-1]:.4f}")
    print(f"  Val   loss: {mlp['val_loss'][0]:.4f} -> {mlp['val_loss'][-1]:.4f}")
    print(f"  Best val loss: {best_mlp_val:.4f} at epoch {mlp_epochs[best_mlp_idx]}")
    print(f"  Best val AUC: {best_auc_mlp:.4f} at epoch {mlp_epochs[best_auc_mlp_idx]}")
    print(f"  LR: {mlp['lr'][0]:.6f} -> {mlp['lr'][-1]:.2e} (cosine annealing)")


if __name__ == "__main__":
    main()
