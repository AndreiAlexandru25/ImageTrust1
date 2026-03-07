#!/usr/bin/env python
"""
Generate Train vs. Test ROC Curves — Overfitting Diagnostic.

Produces fig11_train_test_roc.pdf (2x3 grid, one panel per model) showing
train ROC (dashed) vs. test ROC (solid) with AUC delta for overfitting
assessment.

Pipeline:
  1. Load 414 embedding shards (same as Phase 3)
  2. Recreate exact Phase 2 splits (seed=42, 70/15/15, stratified by image_id)
  3. For each of 6 models:
     - Baselines (LogReg, 3x XGBoost): retrain on train set, predict train+test
     - Meta-classifiers (XGBoost, MLP): load saved models, predict on train+test
  4. Generate 2x3 ROC figure with train/test overlay

Reuses data pipeline from run_phase3_publication.py for exact reproducibility.

Usage:
    python scripts/generate_train_test_roc.py
    python scripts/generate_train_test_roc.py --max_samples 100000  # quick test

Author: ImageTrust Research Team
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# =============================================================================
# CONFIGURATION (matches run_phase3_publication.py exactly)
# =============================================================================

PRIMARY_SEED = 42
BACKBONES = ["resnet50", "efficientnet_b0", "vit_b_16"]
BACKBONE_DIMS = {"resnet50": 2048, "efficientnet_b0": 1280, "vit_b_16": 768}

# Figure settings (publication quality, same as Phase 3)
FIG_DPI = 300
FIG_FORMAT = ["pdf", "png"]
FONT_CONFIG = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "text.usetex": False,
    "mathtext.fontset": "dejavuserif",
}

# Colors (same palette as Phase 3)
COLORS = {
    "logreg": "#7570b3",
    "single_resnet50": "#d95f02",
    "single_efficientnet_b0": "#e7298a",
    "single_vit_b_16": "#66a61e",
    "xgboost_meta": "#1b9e77",
    "mlp_meta": "#e6ab02",
}

# Display names for figure panels
PANEL_TITLES = {
    "logreg": "B1: LogReg (ResNet-50)",
    "single_resnet50": "B2: XGBoost (ResNet-50)",
    "single_efficientnet_b0": "B2: XGBoost (EfficientNet-B0)",
    "single_vit_b_16": "B3: XGBoost (ViT-B/16)",
    "xgboost_meta": "Ours: XGBoost Meta (3-backbone)",
    "mlp_meta": "Ours: MLP Meta (3-backbone)",
}

# Model ordering for 2x3 grid
MODEL_ORDER = [
    "logreg",
    "single_resnet50",
    "single_efficientnet_b0",
    "single_vit_b_16",
    "xgboost_meta",
    "mlp_meta",
]


# =============================================================================
# UTILITIES
# =============================================================================

def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
    print(f"[{ts}] [{level:5s}] {safe_msg}", flush=True)


def set_seeds(seed: int):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def correct_label_from_image_id(image_id: str) -> int:
    """Infer ground-truth label from image_id prefix. 0=real, 1=AI-generated."""
    s = image_id.lower()
    if s.startswith("cifake_real_"):
        return 0
    if s.startswith("coco_"):
        return 0
    if s.startswith("ffhq_"):
        return 0
    if s.startswith("other_real_"):
        return 0
    if s.startswith("deepfake_deepfake and real images_real"):
        return 0
    if s.startswith("deepfake_deepfake-and-real-images_real"):
        return 0
    if s.startswith("cifake_sd_"):
        return 1
    if s.startswith("sfhq_"):
        return 1
    if s.startswith("deepfake_faces_"):
        return 1
    if s.startswith("hard_fakes_"):
        return 1
    if "fake" in s or "generated" in s or "synthetic" in s:
        return 1
    return 0


# =============================================================================
# DATA LOADING (mirrors run_phase3_publication.py exactly)
# =============================================================================

def load_embeddings_and_split(
    embeddings_dir: Path, max_samples: int = 0
) -> Dict[str, Any]:
    """Load Phase 1 embeddings and recreate Phase 2 splits."""
    from sklearn.model_selection import train_test_split

    index_path = embeddings_dir / "embedding_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Embedding index not found: {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    log(f"Loading embeddings from {index['total_shards']} shards...")
    backbones = index["backbones"]
    emb_lists = {b: [] for b in backbones}
    quality_list, labels_list, ids_list, variants_list = [], [], [], []
    samples_loaded = 0

    total_shards = len(index["shard_files"])
    if max_samples > 0 and index["total_samples"] > max_samples:
        avg_per_shard = index["total_samples"] / total_shards
        needed_shards = min(
            total_shards, int(np.ceil(max_samples / avg_per_shard)) + 5
        )
        shard_indices = np.linspace(
            0, total_shards - 1, needed_shards, dtype=int
        ).tolist()
    else:
        shard_indices = list(range(total_shards))

    for count, i in enumerate(shard_indices):
        if max_samples > 0 and samples_loaded >= max_samples:
            break
        shard_file = index["shard_files"][i]
        shard_path = embeddings_dir / shard_file
        if not shard_path.exists():
            continue
        shard = np.load(shard_path, allow_pickle=True)
        n = len(shard["labels"])
        if max_samples > 0:
            n = min(n, max_samples - samples_loaded)
        for b in backbones:
            emb_lists[b].append(shard[f"embeddings_{b}"][:n])
        quality_list.append(shard["niqe_scores"][:n].astype(np.float32))
        shard_ids = shard["image_ids"][:n]
        corrected_labels = np.array(
            [correct_label_from_image_id(str(iid)) for iid in shard_ids],
            dtype=np.int32,
        )
        labels_list.append(corrected_labels)
        ids_list.append(shard_ids)
        variants_list.append(shard["variant_types"][:n])
        samples_loaded += n
        shard.close()
        if (count + 1) % 100 == 0:
            log(f"  {count+1}/{len(shard_indices)} shards "
                f"({samples_loaded:,} samples)")

    log(f"Loaded {samples_loaded:,} samples, concatenating...")
    embeddings = {}
    for b in backbones:
        embeddings[b] = np.vstack(emb_lists[b])
    del emb_lists
    quality = np.concatenate(quality_list)
    labels = np.concatenate(labels_list)
    image_ids = np.concatenate(ids_list)
    variant_types = np.concatenate(variants_list)
    del quality_list, labels_list, ids_list, variants_list
    gc.collect()

    # Recreate exact same splits as Phase 2
    unique_ids = np.unique(image_ids)
    id_to_label = dict(zip(image_ids, labels))
    unique_labels = np.array([id_to_label[uid] for uid in unique_ids])

    train_ids, valtest_ids, _, valtest_labels = train_test_split(
        unique_ids, unique_labels,
        test_size=0.30, stratify=unique_labels, random_state=PRIMARY_SEED,
    )
    val_ids, test_ids = train_test_split(
        valtest_ids, test_size=0.50,
        stratify=valtest_labels, random_state=PRIMARY_SEED,
    )

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    train_mask = np.array([iid in train_set for iid in image_ids])
    val_mask = np.array([iid in val_set for iid in image_ids])
    test_mask = np.array([iid in test_set for iid in image_ids])

    log(f"Split: Train={train_mask.sum():,}, Val={val_mask.sum():,}, "
        f"Test={test_mask.sum():,}")

    return {
        "embeddings": embeddings,
        "quality": quality,
        "labels": labels,
        "image_ids": image_ids,
        "variant_types": variant_types,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


# =============================================================================
# MODEL TRAINING / PREDICTION
# =============================================================================

def get_train_test_predictions(
    data: Dict[str, Any],
    phase2_dir: Path,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Generate train + test predictions for all 6 models.

    Returns dict: {model_name: {"train_proba", "train_labels",
                                 "test_proba", "test_labels"}}
    """
    embeddings = data["embeddings"]
    quality = data["quality"]
    labels = data["labels"]
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    results = {}

    def batched_predict_proba(model, X, batch_size=100_000):
        """Predict in batches to avoid OOM on large arrays."""
        probas = []
        for i in range(0, len(X), batch_size):
            probas.append(model.predict_proba(X[i:i + batch_size])[:, 1])
        return np.concatenate(probas)

    # ── 1. LogReg on ResNet-50 embeddings ──
    log("Training LogReg on ResNet-50 embeddings...")
    set_seeds(PRIMARY_SEED)
    from sklearn.linear_model import LogisticRegression

    X_train_lr = embeddings["resnet50"][train_mask].astype(np.float32)
    X_test_lr = embeddings["resnet50"][test_mask].astype(np.float32)
    y_train_lr = labels[train_mask]
    y_test_lr = labels[test_mask]

    lr_model = LogisticRegression(
        C=1.0, max_iter=200, solver="saga", random_state=PRIMARY_SEED, n_jobs=-1,
    )
    lr_model.fit(X_train_lr, y_train_lr)
    results["logreg"] = {
        "train_proba": batched_predict_proba(lr_model, X_train_lr),
        "train_labels": y_train_lr,
        "test_proba": batched_predict_proba(lr_model, X_test_lr),
        "test_labels": y_test_lr,
    }
    log(f"  LogReg done: train AUC will be computed later")
    del X_train_lr, X_test_lr, lr_model
    gc.collect()

    # ── 2-4. Single-backbone XGBoost baselines ──
    import xgboost as xgb

    for backbone in BACKBONES:
        bname = f"single_{backbone}"
        log(f"Training {bname} (XGBoost on {backbone} + NIQE)...")
        set_seeds(PRIMARY_SEED)

        X_train = np.hstack([
            embeddings[backbone][train_mask].astype(np.float32),
            quality[train_mask].reshape(-1, 1),
        ])
        X_val = np.hstack([
            embeddings[backbone][val_mask].astype(np.float32),
            quality[val_mask].reshape(-1, 1),
        ])
        X_test = np.hstack([
            embeddings[backbone][test_mask].astype(np.float32),
            quality[test_mask].reshape(-1, 1),
        ])
        y_train = labels[train_mask]
        y_val = labels[val_mask]
        y_test = labels[test_mask]

        model = xgb.XGBClassifier(
            n_estimators=1000, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            tree_method="hist", device="cuda",
            early_stopping_rounds=50, random_state=PRIMARY_SEED,
            eval_metric="logloss", verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        results[bname] = {
            "train_proba": batched_predict_proba(model, X_train),
            "train_labels": y_train,
            "test_proba": batched_predict_proba(model, X_test),
            "test_labels": y_test,
        }
        log(f"  {bname} done")
        del X_train, X_val, X_test, model
        gc.collect()

    # ── 5. XGBoost meta-classifier (load saved model) ──
    log("Loading XGBoost meta-classifier...")
    xgb_model_path = phase2_dir / "xgboost" / "meta_classifier.xgb"
    if xgb_model_path.exists():
        meta_xgb = xgb.XGBClassifier()
        meta_xgb.load_model(str(xgb_model_path))

        # Build 4097-d meta-features (all backbones + NIQE)
        X_train_meta = np.hstack([
            embeddings[b][train_mask].astype(np.float32) for b in BACKBONES
        ] + [quality[train_mask].reshape(-1, 1)])
        X_test_meta = np.hstack([
            embeddings[b][test_mask].astype(np.float32) for b in BACKBONES
        ] + [quality[test_mask].reshape(-1, 1)])

        results["xgboost_meta"] = {
            "train_proba": batched_predict_proba(meta_xgb, X_train_meta),
            "train_labels": labels[train_mask],
            "test_proba": batched_predict_proba(meta_xgb, X_test_meta),
            "test_labels": labels[test_mask],
        }
        log("  XGBoost meta done")
    else:
        log(f"  XGBoost meta model not found: {xgb_model_path}", "WARN")

    # ── 6. MLP meta-classifier (load saved model) ──
    log("Loading MLP meta-classifier...")
    mlp_path = phase2_dir / "mlp" / "meta_classifier_seed42.pt"
    if mlp_path.exists():
        import torch
        import torch.nn as nn

        # Build MLP with exact Phase 2 architecture: GELU activation, 1 output
        hidden_dims = [1024, 512, 256]
        input_dim = 4097
        dropout = 0.3

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h), nn.BatchNorm1d(h),
                nn.GELU(), nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        mlp_model = nn.Sequential(*layers)

        state = torch.load(mlp_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            mlp_model.load_state_dict(state["state_dict"])
        else:
            mlp_model.load_state_dict(state)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mlp_model = mlp_model.to(device).eval()

        X_train_meta = np.hstack([
            embeddings[b][train_mask].astype(np.float32) for b in BACKBONES
        ] + [quality[train_mask].reshape(-1, 1)])
        X_test_meta = np.hstack([
            embeddings[b][test_mask].astype(np.float32) for b in BACKBONES
        ] + [quality[test_mask].reshape(-1, 1)])

        # Predict in batches to avoid OOM
        def predict_mlp(model, X, device, batch_size=4096):
            probas = []
            for i in range(0, len(X), batch_size):
                batch = torch.tensor(
                    X[i:i + batch_size], dtype=torch.float32
                ).to(device)
                with torch.no_grad():
                    logits = model(batch).squeeze(-1)
                    proba = torch.sigmoid(logits).cpu().numpy()
                probas.append(proba)
            return np.concatenate(probas)

        results["mlp_meta"] = {
            "train_proba": predict_mlp(mlp_model, X_train_meta, device),
            "train_labels": labels[train_mask],
            "test_proba": predict_mlp(mlp_model, X_test_meta, device),
            "test_labels": labels[test_mask],
        }
        log("  MLP meta done")
        del mlp_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        log(f"  MLP model not found: {mlp_path}", "WARN")

    return results


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_figure(
    predictions: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
):
    """Generate the 2x3 train vs. test ROC figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc as sk_auc

    plt.rcParams.update(FONT_CONFIG)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes_flat = axes.flatten()

    for idx, model_name in enumerate(MODEL_ORDER):
        ax = axes_flat[idx]

        if model_name not in predictions:
            ax.set_title(PANEL_TITLES[model_name] + " (N/A)")
            ax.text(0.5, 0.5, "Model not available",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        preds = predictions[model_name]
        color = COLORS[model_name]

        # Test ROC (solid)
        fpr_test, tpr_test, _ = roc_curve(
            preds["test_labels"], preds["test_proba"]
        )
        auc_test = sk_auc(fpr_test, tpr_test)

        # Train ROC (dashed)
        fpr_train, tpr_train, _ = roc_curve(
            preds["train_labels"], preds["train_proba"]
        )
        auc_train = sk_auc(fpr_train, tpr_train)

        delta = auc_train - auc_test

        ax.plot(fpr_test, tpr_test, color=color, linewidth=2.0,
                label=f"Test (AUC={auc_test:.3f})")
        ax.plot(fpr_train, tpr_train, color=color, linewidth=1.5,
                linestyle="--", alpha=0.6,
                label=f"Train (AUC={auc_train:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)

        # Delta annotation
        ax.text(0.97, 0.03, f"$\\Delta$ = {delta:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8))

        ax.set_title(PANEL_TITLES[model_name], fontsize=11, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.legend(loc="lower right", framealpha=0.9)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Train vs. Test ROC Curves \u2014 Overfitting Diagnostic",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for fmt in FIG_FORMAT:
        path = fig_dir / f"fig11_train_test_roc.{fmt}"
        fig.savefig(str(path), format=fmt, dpi=FIG_DPI, bbox_inches="tight")
        log(f"  Saved: {path}")
    plt.close(fig)


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_test_predictions(
    predictions: Dict[str, Dict[str, np.ndarray]],
    baselines_dir: Path,
):
    """Sanity check: verify retrained baseline test predictions match saved ones."""
    from sklearn.metrics import roc_auc_score

    log("Verifying test predictions against saved baselines...")
    mismatches = 0

    for bname in ["single_resnet50", "single_efficientnet_b0",
                   "single_vit_b_16", "logreg"]:
        npz_path = baselines_dir / f"{bname}_predictions.npz"
        if not npz_path.exists() or bname not in predictions:
            continue

        saved = np.load(npz_path)
        saved_auc = roc_auc_score(saved["y_test"], saved["y_proba_test"])
        new_auc = roc_auc_score(
            predictions[bname]["test_labels"],
            predictions[bname]["test_proba"],
        )
        diff = abs(saved_auc - new_auc)
        status = "OK" if diff < 0.005 else "MISMATCH"
        if status == "MISMATCH":
            mismatches += 1
        log(f"  {bname}: saved AUC={saved_auc:.4f}, "
            f"retrained AUC={new_auc:.4f}, diff={diff:.4f} [{status}]")

    if mismatches > 0:
        log(f"  WARNING: {mismatches} baseline(s) show AUC diff > 0.005. "
            "This may be due to XGBoost GPU non-determinism.", "WARN")
    else:
        log("  All baselines match saved predictions (within tolerance).")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Train vs. Test ROC Curves"
    )
    parser.add_argument(
        "--embeddings_dir", type=str,
        default="data/phase1/embeddings",
        help="Path to Phase 1 embeddings directory",
    )
    parser.add_argument(
        "--phase2_dir", type=str,
        default="models/phase2",
        help="Path to Phase 2 model directory",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="outputs/phase3",
        help="Output directory for figures and predictions",
    )
    parser.add_argument(
        "--max_samples", type=int, default=604589,
        help="Max samples to load (default 604589, matching Phase 2 training)",
    )
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    phase2_dir = Path(args.phase2_dir)
    output_dir = Path(args.output_dir)

    log("=" * 70)
    log("ImageTrust: Train vs. Test ROC Curve Generation")
    log("=" * 70)
    t_start = time.time()

    # Step 1: Load embeddings and recreate splits
    log("Step 1: Loading embeddings and recreating splits...")
    data = load_embeddings_and_split(embeddings_dir, args.max_samples)

    # Step 2: Generate train + test predictions for all models
    log("Step 2: Generating train + test predictions for 6 models...")
    predictions = get_train_test_predictions(data, phase2_dir)

    # Step 3: Verify test predictions match saved baselines
    baselines_dir = output_dir / "baselines"
    if baselines_dir.exists():
        verify_test_predictions(predictions, baselines_dir)

    # Step 4: Save train predictions
    train_pred_dir = output_dir / "train_predictions"
    train_pred_dir.mkdir(parents=True, exist_ok=True)
    for model_name, preds in predictions.items():
        np.savez_compressed(
            train_pred_dir / f"{model_name}_train_predictions.npz",
            train_proba=preds["train_proba"],
            train_labels=preds["train_labels"],
            test_proba=preds["test_proba"],
            test_labels=preds["test_labels"],
        )
    log(f"Saved train predictions to {train_pred_dir}")

    # Step 5: Print AUC summary
    from sklearn.metrics import roc_auc_score
    log("\n" + "=" * 70)
    log("AUC Summary (Train vs. Test)")
    log("=" * 70)
    log(f"{'Model':<35s} {'Train AUC':>10s} {'Test AUC':>10s} {'Delta':>8s}")
    log("-" * 70)
    for model_name in MODEL_ORDER:
        if model_name not in predictions:
            continue
        preds = predictions[model_name]
        auc_train = roc_auc_score(preds["train_labels"], preds["train_proba"])
        auc_test = roc_auc_score(preds["test_labels"], preds["test_proba"])
        delta = auc_train - auc_test
        log(f"{PANEL_TITLES[model_name]:<35s} {auc_train:>10.4f} "
            f"{auc_test:>10.4f} {delta:>+8.4f}")

    # Step 6: Generate figure
    log("\nStep 6: Generating figure...")
    generate_figure(predictions, output_dir)

    # Cleanup
    del data, predictions
    gc.collect()

    elapsed = time.time() - t_start
    log(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log(f"Output: {output_dir / 'figures' / 'fig11_train_test_roc.pdf'}")


if __name__ == "__main__":
    main()
