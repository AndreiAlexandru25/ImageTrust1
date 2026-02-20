#!/usr/bin/env python
"""
ImageTrust v2.0 - Cross-Generator Evaluation Pipeline.

Evaluates the Phase 2 meta-classifiers on 33 generators from the artifacts
dataset. Produces Table 2 (per-generator breakdown) and cross-generator
heatmap data for the paper.

Pipeline:
1. Sample up to N images per generator (default: 5000)
2. Extract embeddings with 3 backbones (ResNet50, EfficientNet-B0, ViT-B/16)
3. Evaluate with Phase 2 XGBoost and MLP models
4. Generate per-generator metrics + LaTeX table + heatmap data

Input:  data/extracted/artifacts/ (33 generator directories)
Output: models/phase2/cross_generator/ (metrics, tables, figures data)

Usage:
    python scripts/orchestrator/run_cross_generator_eval.py
    python scripts/orchestrator/run_cross_generator_eval.py --max_per_gen 2000
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# ── Configuration ──

# Real datasets (label=0) - these contain authentic photographs
REAL_GENERATORS = {
    "afhq",          # Animal Faces HQ (real animal photos)
    "celebahq",      # CelebA-HQ (real celebrity faces)
    "coco",          # MS COCO (real diverse photos)
    "ffhq",          # Flickr Faces HQ (real faces)
    "imagenet",      # ImageNet (real diverse photos)
    "landscape",     # Real landscape photos
    "lsun",          # Large-Scale Unseen (real scene photos)
    "metfaces",      # Metropolitan Museum faces (real artwork)
    "sfhq",          # Synthetic Faces HQ - actually real reference set
}

# AI-generated datasets (label=1) - all other generators
AI_GENERATORS = {
    "big_gan", "cips", "cycle_gan", "ddpm", "denoising_diffusion_gan",
    "diffusion_gan", "face_synthetics", "gansformer", "gau_gan",
    "generative_inpainting", "glide", "lama", "latent_diffusion",
    "mat", "palette", "pro_gan", "projected_gan", "stable_diffusion",
    "star_gan", "stylegan1", "stylegan2", "stylegan3",
    "taming_transformer", "vq_diffusion",
}

# Generator display names for paper
GENERATOR_NAMES = {
    "afhq": "AFHQ (Real)",
    "big_gan": "BigGAN",
    "celebahq": "CelebA-HQ (Real)",
    "cips": "CIPS",
    "coco": "COCO (Real)",
    "cycle_gan": "CycleGAN",
    "ddpm": "DDPM",
    "denoising_diffusion_gan": "Denoising Diff. GAN",
    "diffusion_gan": "DiffusionGAN",
    "face_synthetics": "Face Synthetics",
    "ffhq": "FFHQ (Real)",
    "gansformer": "GANsformer",
    "gau_gan": "GauGAN",
    "generative_inpainting": "Gen. Inpainting",
    "glide": "GLIDE",
    "imagenet": "ImageNet (Real)",
    "lama": "LaMa",
    "landscape": "Landscape (Real)",
    "latent_diffusion": "Latent Diffusion",
    "lsun": "LSUN (Real)",
    "mat": "MAT",
    "metfaces": "MetFaces (Real)",
    "palette": "Palette",
    "pro_gan": "ProGAN",
    "projected_gan": "ProjectedGAN",
    "sfhq": "SFHQ (Real)",
    "stable_diffusion": "Stable Diffusion",
    "star_gan": "StarGAN",
    "stylegan1": "StyleGAN",
    "stylegan2": "StyleGAN2",
    "stylegan3": "StyleGAN3",
    "taming_transformer": "Taming Transf.",
    "vq_diffusion": "VQ-Diffusion",
}

# Generator categories for grouping
GENERATOR_CATEGORIES = {
    "GAN": ["big_gan", "cips", "cycle_gan", "diffusion_gan", "gansformer",
            "gau_gan", "pro_gan", "projected_gan", "star_gan",
            "stylegan1", "stylegan2", "stylegan3"],
    "Diffusion": ["ddpm", "denoising_diffusion_gan", "glide",
                  "latent_diffusion", "palette", "stable_diffusion",
                  "taming_transformer", "vq_diffusion"],
    "Inpainting": ["generative_inpainting", "lama", "mat"],
    "Other AI": ["face_synthetics"],
    "Real": list(REAL_GENERATORS),
}

BACKBONES = ["resnet50", "efficientnet_b0", "vit_b_16"]
BACKBONE_DIMS = {"resnet50": 2048, "efficientnet_b0": 1280, "vit_b_16": 768}

SEED = 42


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


# ── Image Loading ──

def find_images(directory: Path, max_count: int = 5000) -> List[Path]:
    """Find image files in directory, sample up to max_count."""
    extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    images = []
    for ext in extensions:
        images.extend(directory.rglob(f"*{ext}"))
        images.extend(directory.rglob(f"*{ext.upper()}"))
    images = sorted(set(images))

    if len(images) > max_count:
        random.seed(SEED)
        images = random.sample(images, max_count)
    return images


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            tensor = self.transform(image)
            return {"tensor": tensor, "path": str(path), "success": True}
        except Exception:
            return {
                "tensor": torch.zeros(3, 224, 224),
                "path": str(path),
                "success": False,
            }


# ── Backbone Embedder ──

class BackboneEmbedder:
    """Extract embeddings from a backbone model."""

    def __init__(self, backbone_name: str, device: str = "cuda"):
        self.backbone_name = backbone_name
        self.device = device
        self.embed_dim = BACKBONE_DIMS[backbone_name]

        self.model = self._load_backbone()
        self.model.to(device)
        self.model.eval()

        self._embeddings = None
        self._register_hook()

    def _load_backbone(self) -> nn.Module:
        import torchvision.models as models
        if self.backbone_name == "resnet50":
            return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif self.backbone_name == "efficientnet_b0":
            return models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
        elif self.backbone_name == "vit_b_16":
            return models.vit_b_16(
                weights=models.ViT_B_16_Weights.IMAGENET1K_V1
            )
        raise ValueError(f"Unknown backbone: {self.backbone_name}")

    def _register_hook(self):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self._embeddings = output.detach()
            elif isinstance(output, tuple):
                self._embeddings = output[0].detach()

        if self.backbone_name == "resnet50":
            self.model.avgpool.register_forward_hook(hook_fn)
        elif self.backbone_name == "efficientnet_b0":
            self.model.avgpool.register_forward_hook(hook_fn)
        elif self.backbone_name == "vit_b_16":
            self.model.encoder.ln.register_forward_hook(hook_fn)

    @torch.no_grad()
    def extract(self, batch: torch.Tensor) -> np.ndarray:
        batch = batch.to(self.device)
        with torch.amp.autocast("cuda"):
            self.model(batch)
        emb = self._embeddings.float()
        if emb.dim() == 4:  # CNN: (B, C, 1, 1)
            emb = emb.squeeze(-1).squeeze(-1)
        elif emb.dim() == 3:  # ViT: (B, seq_len, dim) -> take CLS token
            emb = emb[:, 0, :]
        return emb.cpu().numpy().astype(np.float16)


# ── Main Pipeline ──

def extract_embeddings_for_generator(
    gen_dir: Path,
    gen_name: str,
    max_images: int,
    embedders: Dict[str, BackboneEmbedder],
    transform,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Optional[Dict[str, np.ndarray]]:
    """Extract embeddings for one generator directory."""
    images = find_images(gen_dir, max_images)
    if not images:
        log(f"  {gen_name}: no images found, skipping", "WARN")
        return None

    dataset = ImageDataset(images, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    all_embeddings = {b: [] for b in embedders}
    n_success = 0

    for batch_data in loader:
        mask = batch_data["success"]
        tensors = batch_data["tensor"][mask]
        if tensors.shape[0] == 0:
            continue

        for backbone_name, embedder in embedders.items():
            emb = embedder.extract(tensors)
            all_embeddings[backbone_name].append(emb)
        n_success += tensors.shape[0]

    if n_success == 0:
        return None

    result = {}
    for backbone_name in embedders:
        result[backbone_name] = np.concatenate(
            all_embeddings[backbone_name], axis=0
        )

    log(f"  {gen_name}: {n_success} images embedded")
    return result


def evaluate_generator(
    embeddings: Dict[str, np.ndarray],
    label: int,
    xgb_model,
    mlp_model,
    mlp_device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate a generator's embeddings with Phase 2 models."""
    # Concatenate features (same as Phase 2 training)
    parts = [embeddings[b] for b in BACKBONES if b in embeddings]
    X = np.hstack(parts).astype(np.float32)

    # Add dummy NIQE column (0.0) since we don't compute NIQE here
    niqe_col = np.zeros((X.shape[0], 1), dtype=np.float32)
    X = np.hstack([X, niqe_col])

    # Build feature names matching Phase 2 training format
    feature_names = []
    for b in BACKBONES:
        if b in embeddings:
            dim = BACKBONE_DIMS[b]
            feature_names.extend([f"{b}_{i}" for i in range(dim)])
    feature_names.append("niqe")

    n_samples = X.shape[0]
    y_true = np.full(n_samples, label, dtype=np.int32)

    results = {}

    # XGBoost evaluation
    if xgb_model is not None:
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        y_proba = xgb_model.predict(dmatrix)
        y_pred = (y_proba >= 0.5).astype(int)
        acc = float(np.mean(y_pred == y_true))
        # For single-class evaluation, compute detection rate
        if label == 1:  # AI-generated
            detection_rate = float(np.mean(y_pred == 1))  # TPR
        else:  # Real
            detection_rate = float(np.mean(y_pred == 0))  # TNR
        mean_prob = float(np.mean(y_proba))
        results["xgboost"] = {
            "accuracy": acc,
            "detection_rate": detection_rate,
            "mean_probability": mean_prob,
            "n_samples": n_samples,
            "predictions": y_proba.tolist(),
        }

    # MLP evaluation
    if mlp_model is not None:
        mlp_model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(mlp_device)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                logits = mlp_model(X_tensor).squeeze(-1)
                y_proba_mlp = torch.sigmoid(logits).cpu().numpy()
        y_pred_mlp = (y_proba_mlp >= 0.5).astype(int)
        acc_mlp = float(np.mean(y_pred_mlp == y_true))
        if label == 1:
            detection_rate_mlp = float(np.mean(y_pred_mlp == 1))
        else:
            detection_rate_mlp = float(np.mean(y_pred_mlp == 0))
        mean_prob_mlp = float(np.mean(y_proba_mlp))
        results["mlp"] = {
            "accuracy": acc_mlp,
            "detection_rate": detection_rate_mlp,
            "mean_probability": mean_prob_mlp,
            "n_samples": n_samples,
            "predictions": y_proba_mlp.tolist(),
        }

    return results


def load_phase2_models(phase2_dir: Path):
    """Load trained Phase 2 XGBoost and MLP models."""
    xgb_model = None
    mlp_model = None

    # Load XGBoost
    xgb_path = phase2_dir / "xgboost" / "meta_classifier.xgb"
    if xgb_path.exists():
        import xgboost as xgb
        xgb_model = xgb.Booster()
        xgb_model.load_model(str(xgb_path))
        log(f"  Loaded XGBoost from {xgb_path}")
    else:
        log(f"  XGBoost model not found at {xgb_path}", "WARN")

    # Load MLP (best seed = seed42 typically)
    mlp_dir = phase2_dir / "mlp"
    mlp_path = mlp_dir / "meta_classifier_seed42.pt"
    if mlp_path.exists():
        checkpoint = torch.load(mlp_path, map_location="cpu", weights_only=False)

        # Get architecture from checkpoint metadata
        input_dim = checkpoint.get("input_dim", 4097)
        hidden_dims = checkpoint.get("hidden_dims", [1024, 512, 256])
        dropout = checkpoint.get("dropout", 0.3)

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        mlp_model = nn.Sequential(*layers)

        # Load state dict
        state_dict = checkpoint.get(
            "state_dict", checkpoint.get("model_state_dict", checkpoint)
        )
        mlp_model.load_state_dict(state_dict)

        mlp_model.cuda()
        mlp_model.eval()
        log(f"  Loaded MLP from {mlp_path} "
            f"(dim={input_dim}, hidden={hidden_dims})")
    else:
        log(f"  MLP model not found at {mlp_path}", "WARN")

    return xgb_model, mlp_model


def generate_latex_cross_generator_table(
    results: Dict[str, Dict], output_path: Path
):
    """Generate LaTeX Table 2: Cross-generator evaluation."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Cross-generator evaluation. Detection rate (\%) for each "
        r"generator, evaluated on models trained without that generator's data. "
        r"Higher is better for AI generators (TPR), higher is better for real "
        r"datasets (TNR).}",
        r"\label{tab:cross_generator}",
        r"\begin{tabular}{l l r r r}",
        r"\toprule",
        r"Category & Generator & $N$ & XGBoost (\%) & MLP (\%) \\",
        r"\midrule",
    ]

    # Group by category
    for category, gen_list in GENERATOR_CATEGORIES.items():
        lines.append(r"\multicolumn{5}{l}{\textit{" + category + r"}} \\")
        for gen_name in sorted(gen_list):
            if gen_name not in results:
                continue
            gen_data = results[gen_name]
            display_name = GENERATOR_NAMES.get(gen_name, gen_name)
            n_samples = gen_data.get("n_samples", 0)

            xgb_rate = gen_data.get("xgboost", {}).get("detection_rate", 0) * 100
            mlp_rate = gen_data.get("mlp", {}).get("detection_rate", 0) * 100

            lines.append(
                f"  & {display_name} & {n_samples:,} & "
                f"{xgb_rate:.1f} & {mlp_rate:.1f} \\\\"
            )
        lines.append(r"\midrule")

    # Remove last \midrule and replace with \bottomrule
    lines[-1] = r"\bottomrule"

    lines.extend([
        r"\end{tabular}",
        r"\end{table*}",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"  LaTeX cross-generator table saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Generator Evaluation Pipeline"
    )
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        default=Path("data/extracted/artifacts"),
    )
    parser.add_argument(
        "--phase2_dir",
        type=Path,
        default=Path("models/phase2"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models/phase2/cross_generator"),
    )
    parser.add_argument(
        "--max_per_gen",
        type=int,
        default=5000,
        help="Max images per generator (default: 5000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    # Set Windows process priority
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000
        )
        log("Process priority: BELOW_NORMAL")
    except Exception:
        pass

    log("=" * 70)
    log("ImageTrust v2.0 -- Cross-Generator Evaluation")
    log("=" * 70)

    artifacts_dir = Path(args.artifacts_dir)
    phase2_dir = Path(args.phase2_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover generators
    gen_dirs = sorted([
        d for d in artifacts_dir.iterdir()
        if d.is_dir() and d.name != "__MACOSX"
    ])
    log(f"Found {len(gen_dirs)} generator directories")
    log(f"Max images per generator: {args.max_per_gen}")

    # Check CUDA
    if not torch.cuda.is_available():
        log("CUDA not available! This requires GPU.", "ERROR")
        sys.exit(1)
    log(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load Phase 2 models
    log("\nLoading Phase 2 models...")
    xgb_model, mlp_model = load_phase2_models(phase2_dir)

    # Load backbones
    log("\nLoading backbone models...")
    embedders = {}
    for backbone in BACKBONES:
        log(f"  Loading {backbone}...")
        embedders[backbone] = BackboneEmbedder(backbone, "cuda")

    # Image transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Process each generator
    log(f"\nProcessing {len(gen_dirs)} generators...")
    all_results = {}
    pipeline_start = time.time()

    for i, gen_dir in enumerate(gen_dirs):
        gen_name = gen_dir.name
        label = 0 if gen_name in REAL_GENERATORS else 1
        label_str = "REAL" if label == 0 else "AI"

        log(f"\n[{i+1}/{len(gen_dirs)}] {gen_name} ({label_str})")

        # Extract embeddings
        t0 = time.time()
        embeddings = extract_embeddings_for_generator(
            gen_dir, gen_name, args.max_per_gen,
            embedders, transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        if embeddings is None:
            log(f"  Skipped {gen_name}: no valid images")
            continue

        # Evaluate
        eval_results = evaluate_generator(
            embeddings, label, xgb_model, mlp_model
        )

        n_samples = embeddings[BACKBONES[0]].shape[0]
        gen_result = {
            "generator": gen_name,
            "display_name": GENERATOR_NAMES.get(gen_name, gen_name),
            "label": label,
            "label_str": label_str,
            "n_samples": n_samples,
            "category": next(
                (cat for cat, gens in GENERATOR_CATEGORIES.items()
                 if gen_name in gens),
                "Unknown"
            ),
        }
        gen_result.update(eval_results)

        # Log results
        xgb_rate = eval_results.get("xgboost", {}).get("detection_rate", 0)
        mlp_rate = eval_results.get("mlp", {}).get("detection_rate", 0)
        elapsed = time.time() - t0
        log(f"  {gen_name}: XGB={xgb_rate*100:.1f}% MLP={mlp_rate*100:.1f}% "
            f"({elapsed:.0f}s, {n_samples} imgs)")

        all_results[gen_name] = gen_result

        # Free memory
        del embeddings
        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ──
    pipeline_time = time.time() - pipeline_start

    log("\n" + "=" * 70)
    log("CROSS-GENERATOR EVALUATION SUMMARY")
    log("=" * 70)

    # Print summary table
    log(f"\n  {'Generator':<25} {'Type':<6} {'N':>6} {'XGB%':>7} {'MLP%':>7}")
    log(f"  {'-'*55}")

    for gen_name in sorted(all_results.keys()):
        r = all_results[gen_name]
        xgb_rate = r.get("xgboost", {}).get("detection_rate", 0) * 100
        mlp_rate = r.get("mlp", {}).get("detection_rate", 0) * 100
        log(f"  {GENERATOR_NAMES.get(gen_name, gen_name):<25} "
            f"{r['label_str']:<6} {r['n_samples']:>6} "
            f"{xgb_rate:>6.1f} {mlp_rate:>6.1f}")

    # Category averages
    log(f"\n  Category averages:")
    for category, gen_list in GENERATOR_CATEGORIES.items():
        cat_xgb = [
            all_results[g]["xgboost"]["detection_rate"]
            for g in gen_list if g in all_results and "xgboost" in all_results[g]
        ]
        cat_mlp = [
            all_results[g]["mlp"]["detection_rate"]
            for g in gen_list if g in all_results and "mlp" in all_results[g]
        ]
        if cat_xgb:
            log(f"    {category:<15} XGB={np.mean(cat_xgb)*100:.1f}% "
                f"MLP={np.mean(cat_mlp)*100:.1f}%")

    # Save results (without per-image predictions for space)
    results_clean = {}
    for gen_name, r in all_results.items():
        r_clean = {k: v for k, v in r.items()}
        for model_name in ["xgboost", "mlp"]:
            if model_name in r_clean:
                r_clean[model_name] = {
                    k: v for k, v in r_clean[model_name].items()
                    if k != "predictions"
                }
        results_clean[gen_name] = r_clean

    with open(output_dir / "cross_generator_results.json", "w") as f:
        json.dump(results_clean, f, indent=2)

    # Generate LaTeX table
    generate_latex_cross_generator_table(
        all_results, output_dir / "table_cross_generator.tex"
    )

    # Generate heatmap data (for matplotlib figure)
    heatmap_data = {
        "generators": list(all_results.keys()),
        "display_names": [
            GENERATOR_NAMES.get(g, g) for g in all_results.keys()
        ],
        "labels": [all_results[g]["label"] for g in all_results.keys()],
        "xgboost_rates": [
            all_results[g].get("xgboost", {}).get("detection_rate", 0)
            for g in all_results.keys()
        ],
        "mlp_rates": [
            all_results[g].get("mlp", {}).get("detection_rate", 0)
            for g in all_results.keys()
        ],
    }
    with open(output_dir / "heatmap_data.json", "w") as f:
        json.dump(heatmap_data, f, indent=2)

    log(f"\nTotal time: {timedelta(seconds=int(pipeline_time))}")
    log(f"Output: {output_dir}")
    log("Done!")


if __name__ == "__main__":
    main()
