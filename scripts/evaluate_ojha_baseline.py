#!/usr/bin/env python
"""
Ojha et al. (2023) External Baseline Evaluation.

Evaluates the pre-trained UniversalFakeDetect model (CLIP ViT-L/14 + linear layer)
on the exact same test split used for ImageTrust Phase 2 evaluation.

Produces:
- Main comparison metrics (Acc, Prec, Rec, F1, AUC, ECE) for Table 1
- Per-variant degradation metrics (Clean/WhatsApp/Instagram/Screenshot AUC) for Table 3
- JSON results file for integration

Reference: Ojha et al., "Towards Universal Fake Image Detectors that Generalize
           Across Generative Models", CVPR 2023
Model: https://github.com/WisconsinAIVision/UniversalFakeDetect

Usage:
    python scripts/evaluate_ojha_baseline.py

Hardware: RTX 5080 (16GB), AMD 7800X3D, 32GB RAM
Expected runtime: ~45 minutes (CLIP ViT-L/14 is larger than ResNet-50)
"""

import gc
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ============================================================================
# CONFIGURATION
# ============================================================================

EMBEDDINGS_DIR = Path("data/phase1/embeddings")
EXTRACTED_DIR = Path("data/extracted")
OUTPUT_DIR = Path("outputs/ojha_baseline")
MODEL_DIR = Path("models/ojha_et_al")
REPO_DIR = MODEL_DIR / "UniversalFakeDetect"

# Phase 2 split parameters (must match exactly)
PRIMARY_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 32  # Smaller than Wang due to larger CLIP ViT-L/14
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ECE parameters
ECE_BINS = 15

# CLIP normalisation (NOT ImageNet — Ojha uses CLIP stats)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# ============================================================================
# LABEL CORRECTION (from Phase 2 — must be identical)
# ============================================================================

def correct_label_from_image_id(image_id: str) -> int:
    """Correctly infer label from image_id. 0=real, 1=AI."""
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
    if s.startswith("cifake_sd_"):
        return 1
    if s.startswith("sfhq_"):
        return 1
    if s.startswith("deepfake_faces_"):
        return 1
    if s.startswith("hard_fakes_"):
        return 1
    if "fake" in s:
        return 1
    return 0


# ============================================================================
# IMAGE ID GENERATION (from Phase 1 — must be identical)
# ============================================================================

def generate_image_id(image_path: Path) -> str:
    """Generate deterministic unique ID for an image."""
    path_hash = hashlib.md5(str(image_path.resolve()).encode()).hexdigest()[:12]
    return f"{image_path.stem}_{path_hash}"


# ============================================================================
# STAGE 1: REPRODUCE PHASE 2 SPLIT
# ============================================================================

def get_test_image_ids() -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce the Phase 2 split to get test image_ids and labels."""
    from sklearn.model_selection import train_test_split

    print("=" * 70)
    print("STAGE 1: Reproducing Phase 2 split")
    print("=" * 70)

    index_path = EMBEDDINGS_DIR / "embedding_index.json"
    with open(index_path) as f:
        index = json.load(f)

    ids_list, labels_list = [], []
    for i, shard_file in enumerate(index["shard_files"]):
        shard_path = EMBEDDINGS_DIR / shard_file
        if not shard_path.exists():
            continue
        shard = np.load(shard_path, allow_pickle=True)
        shard_ids = shard["image_ids"]
        corrected = np.array(
            [correct_label_from_image_id(str(iid)) for iid in shard_ids],
            dtype=np.int32,
        )
        ids_list.append(shard_ids)
        labels_list.append(corrected)
        shard.close()
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(index['shard_files'])} shards")

    image_ids = np.concatenate(ids_list)
    labels = np.concatenate(labels_list)
    del ids_list, labels_list
    gc.collect()

    print(f"  Total samples: {len(image_ids):,}")

    # Reproduce split
    unique_ids = np.unique(image_ids)
    id_to_label = dict(zip(image_ids, labels))
    unique_labels = np.array([id_to_label[uid] for uid in unique_ids])
    print(f"  Unique images: {len(unique_ids):,}")

    _, valtest_ids, _, valtest_labels = train_test_split(
        unique_ids, unique_labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=unique_labels,
        random_state=PRIMARY_SEED,
    )
    val_ratio_adj = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    _, test_ids = train_test_split(
        valtest_ids,
        test_size=(1 - val_ratio_adj),
        stratify=valtest_labels,
        random_state=PRIMARY_SEED,
    )

    test_labels = np.array([id_to_label[tid] for tid in test_ids])

    n_real = (test_labels == 0).sum()
    n_ai = (test_labels == 1).sum()
    print(f"  Test split: {len(test_ids):,} unique images "
          f"(real={n_real:,}, ai={n_ai:,})")

    return test_ids, test_labels


# ============================================================================
# STAGE 2: MAP IMAGE IDS TO FILE PATHS
# ============================================================================

EXTRACTED_TO_FLAT = [
    ("CIFAKE*/train/FAKE", "cifake_sd"),
    ("CIFAKE*/test/FAKE", "cifake_sd"),
    ("CIFAKE*/train/REAL", "cifake_real"),
    ("CIFAKE*/test/REAL", "cifake_real"),
    ("COCO 2017 dataset", "coco"),
    ("Flickr-Faces-HQ Dataset*", "ffhq"),
    ("Synthetic Faces High Quality*", "sfhq"),
    ("deepfake and real images", "deepfake"),
    ("deepfake and real images", "other_real"),
    ("deepfake_faces", "deepfake_faces"),
    ("deepfake_faces", "other_real"),
    ("Fake-Vs-Real-Faces*/fake", "hard_fakes"),
    ("Fake-Vs-Real-Faces*/Fake", "hard_fakes"),
    ("Fake-Vs-Real-Faces*/real", "other_real"),
    ("Fake-Vs-Real-Faces*/Real", "other_real"),
    ("Real and Fake Face Detection*", "fake_faces"),
    ("Real and Fake Face Detection*", "other_real"),
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def build_id_to_path_mapping(test_ids: np.ndarray) -> Dict[str, Path]:
    """Map test image_ids to actual files in data/extracted/."""
    import glob as glob_mod
    import re

    print("\n" + "=" * 70)
    print("STAGE 2: Mapping image IDs to file paths")
    print("=" * 70)

    test_stem_to_ids = {}
    for tid in test_ids:
        tid_str = str(tid)
        parts = tid_str.rsplit("_", 1)
        if len(parts) == 2 and len(parts[1]) == 12:
            stem = parts[0]
        else:
            stem = tid_str
        if stem not in test_stem_to_ids:
            test_stem_to_ids[stem] = []
        test_stem_to_ids[stem].append(tid_str)

    print(f"  Test stems to find: {len(test_stem_to_ids):,}")

    mapping = {}
    for dir_pattern, dataset_target in EXTRACTED_TO_FLAT:
        full_pattern = str(EXTRACTED_DIR / dir_pattern)
        matched_dirs = glob_mod.glob(full_pattern)

        for matched_dir in matched_dirs:
            matched_path = Path(matched_dir)
            if not matched_path.is_dir():
                continue

            src_dir_name = matched_path.name

            for root, dirs, files in os.walk(str(matched_path)):
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in EXTENSIONS:
                        continue

                    stem_no_ext = os.path.splitext(fname)[0]
                    flat_stem = f"{dataset_target}_{src_dir_name}_{stem_no_ext}"

                    if flat_stem in test_stem_to_ids:
                        fpath = Path(root) / fname
                        for original_id in test_stem_to_ids[flat_stem]:
                            if original_id not in mapping:
                                mapping[original_id] = fpath

    print(f"  Direct matches: {len(mapping):,} / {len(test_ids):,} "
          f"({len(mapping) / len(test_ids) * 100:.1f}%)")

    # Dedup-aware matching
    if len(mapping) < len(test_ids):
        dedup_found = 0
        for stem, id_list in test_stem_to_ids.items():
            for original_id in id_list:
                if original_id in mapping:
                    continue
                match = re.match(r'^(.+)_(\d{1,2})$', stem)
                if match:
                    base_stem = match.group(1)
                    if base_stem in test_stem_to_ids:
                        for base_id in test_stem_to_ids[base_stem]:
                            if base_id in mapping:
                                mapping[original_id] = mapping[base_id]
                                dedup_found += 1
                                break

        if dedup_found > 0:
            print(f"  Dedup matches: +{dedup_found:,} (reusing base image)")

    print(f"  Total matched: {len(mapping):,} / {len(test_ids):,} "
          f"({len(mapping) / len(test_ids) * 100:.1f}%)")

    if len(mapping) < len(test_ids) * 0.5:
        print(f"  WARNING: Low match rate!")

    return mapping


# ============================================================================
# STAGE 3: DOWNLOAD / LOAD OJHA MODEL
# ============================================================================

def download_ojha_weights() -> Path:
    """Download UniversalFakeDetect repo and extract fc_weights.pth."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = MODEL_DIR / "fc_weights.pth"

    if weights_path.exists():
        print(f"  Found existing weights: {weights_path}")
        return weights_path

    print("  Downloading Ojha et al. pretrained weights...")
    print("  Source: WisconsinAIVision/UniversalFakeDetect (CVPR 2023)")

    # Clone repo (shallow) to get weights
    if not REPO_DIR.exists():
        print(f"  Cloning repository to {REPO_DIR}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/WisconsinAIVision/UniversalFakeDetect.git",
                 str(REPO_DIR)],
                check=True, capture_output=True, text=True,
            )
            print("  Clone complete.")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: git clone failed: {e.stderr}")
            print(f"\n  Please clone manually:")
            print(f"  git clone https://github.com/WisconsinAIVision/UniversalFakeDetect.git "
                  f"{REPO_DIR}")
            sys.exit(1)

    # Copy weights out
    repo_weights = REPO_DIR / "pretrained_weights" / "fc_weights.pth"
    if repo_weights.exists():
        import shutil
        shutil.copy2(str(repo_weights), str(weights_path))
        print(f"  Weights copied to: {weights_path}")
        print(f"  Size: {weights_path.stat().st_size / 1e3:.1f} KB")
    else:
        print(f"  ERROR: Weights not found at {repo_weights}")
        print(f"  The repo may use Git LFS. Try:")
        print(f"    cd {REPO_DIR} && git lfs pull")
        sys.exit(1)

    return weights_path


def load_ojha_model() -> nn.Module:
    """Load Ojha et al. CLIP ViT-L/14 + linear layer model."""
    print("\n" + "=" * 70)
    print("STAGE 3: Loading Ojha et al. model (CLIP ViT-L/14)")
    print("=" * 70)

    # Install clip if not available
    try:
        import clip
    except ImportError:
        print("  Installing openai-clip package...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"],
            check=True, capture_output=True, text=True,
        )
        import clip

    weights_path = download_ojha_weights()

    # Load CLIP model
    print("  Loading CLIP ViT-L/14 backbone...")
    clip_model, preprocess = clip.load("ViT-L/14", device="cpu")

    # Build the classifier model (matching their CLIPModel class)
    class OjhaClassifier(nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.model = clip_model
            # ViT-L/14 outputs 768-dim features
            self.fc = nn.Linear(768, 1)

        @torch.no_grad()
        def forward(self, x):
            features = self.model.encode_image(x)
            features = features.float()  # CLIP returns float16 on GPU
            return self.fc(features)

    model = OjhaClassifier(clip_model)

    # Load trained linear layer weights
    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    model.fc.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_fc = sum(p.numel() for p in model.fc.parameters())
    print(f"  Model loaded: {n_params_total / 1e6:.1f}M total parameters")
    print(f"  Trainable FC layer: {n_params_fc:,} parameters (768 -> 1)")
    print(f"  CLIP backbone: frozen (not fine-tuned)")
    print(f"  Device: {DEVICE}")

    return model


# ============================================================================
# STAGE 4: VARIANT TRANSFORMS (matching Phase 1 / Wang baseline)
# ============================================================================

def get_ojha_transform():
    """CLIP preprocessing: CenterCrop(224) + CLIP normalisation."""
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def apply_whatsapp(image: Image.Image) -> Image.Image:
    """Simulate WhatsApp compression (aggressive JPEG + resize)."""
    w, h = image.size
    new_w = min(w, 1600)
    ratio = new_w / w
    new_h = int(h * ratio)
    img = image.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=55)
    buf.seek(0)
    img = Image.open(buf).copy()

    buf2 = io.BytesIO()
    img.save(buf2, "JPEG", quality=60)
    buf2.seek(0)
    return Image.open(buf2).copy()


def apply_instagram(image: Image.Image) -> Image.Image:
    """Simulate Instagram compression (chroma 4:2:0 + moderate JPEG)."""
    w, h = image.size
    new_w = min(w, 1080)
    ratio = new_w / w
    new_h = int(h * ratio)
    img = image.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=72, subsampling=2)
    buf.seek(0)
    img = Image.open(buf).copy()

    buf2 = io.BytesIO()
    img.save(buf2, "JPEG", quality=75, subsampling=2)
    buf2.seek(0)
    return Image.open(buf2).copy()


def apply_screenshot(image: Image.Image) -> Image.Image:
    """Simulate screenshot (slight gamma shift + re-render)."""
    img_array = np.array(image).astype(np.float32) / 255.0
    gamma = np.random.uniform(0.95, 1.05)
    img_array = np.clip(np.power(img_array, gamma), 0, 1)
    img_array = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_array)

    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return Image.open(buf).copy()


VARIANT_TRANSFORMS = {
    "original": lambda x: x,
    "whatsapp": apply_whatsapp,
    "instagram": apply_instagram,
    "screenshot": apply_screenshot,
}


# ============================================================================
# STAGE 5: RUN INFERENCE
# ============================================================================

def run_ojha_inference(
    model: nn.Module,
    id_to_path: Dict[str, Path],
    test_ids: np.ndarray,
    test_labels: np.ndarray,
    variant: str = "original",
) -> Dict[str, np.ndarray]:
    """Run Ojha et al. model on test images with optional variant transform."""
    print(f"\n  Running inference: variant={variant}")

    transform = get_ojha_transform()
    variant_fn = VARIANT_TRANSFORMS[variant]

    probas = []
    labels = []
    skipped = 0

    np.random.seed(42)

    valid_count = sum(1 for img_id in test_ids if img_id in id_to_path)
    pbar = tqdm(total=valid_count, desc=f"    {variant}", unit="img")

    with torch.no_grad():
        batch_tensors = []
        batch_labels = []

        for img_id, label in zip(test_ids, test_labels):
            if img_id not in id_to_path:
                skipped += 1
                continue

            try:
                image = Image.open(id_to_path[img_id]).convert("RGB")
                image = variant_fn(image)
                tensor = transform(image)
                batch_tensors.append(tensor)
                batch_labels.append(label)
            except Exception:
                skipped += 1
                pbar.update(1)
                continue

            if len(batch_tensors) >= BATCH_SIZE:
                batch = torch.stack(batch_tensors).to(DEVICE)
                logits = model(batch).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                probas.extend(probs.tolist())
                labels.extend(batch_labels)
                pbar.update(len(batch_tensors))
                batch_tensors = []
                batch_labels = []

        # Final batch
        if batch_tensors:
            batch = torch.stack(batch_tensors).to(DEVICE)
            logits = model(batch).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            probas.extend(probs.tolist())
            labels.extend(batch_labels)
            pbar.update(len(batch_tensors))

    pbar.close()

    print(f"    Done: {len(probas):,} images processed, {skipped:,} skipped")

    return {
        "y_proba": np.array(probas),
        "y_true": np.array(labels),
    }


# ============================================================================
# STAGE 6: COMPUTE METRICS
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute Acc, Prec, Rec, F1, AUC, ECE."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    )

    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    auc = roc_auc_score(y_true, y_proba) * 100

    ece = compute_ece(y_true, y_proba, n_bins=ECE_BINS)

    return {
        "accuracy": round(acc, 1),
        "precision": round(prec, 1),
        "recall": round(rec, 1),
        "f1": round(f1, 1),
        "auc": round(auc, 1),
        "ece": round(ece, 3),
        "n_samples": len(y_true),
        "n_real": int((y_true == 0).sum()),
        "n_ai": int((y_true == 1).sum()),
    }


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_proba >= bin_boundaries[i]) & (y_proba < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_proba[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Ojha et al. (2023) External Baseline Evaluation")
    print("Towards Universal Fake Image Detectors (UniversalFakeDetect)")
    print("CLIP ViT-L/14 + Linear Layer")
    print("=" * 70)

    # Stage 1: Get test split
    test_ids, test_labels = get_test_image_ids()

    # Stage 2: Map IDs to files
    id_to_path = build_id_to_path_mapping(test_ids)

    if len(id_to_path) == 0:
        print("\nERROR: No test images found on disk!")
        print("Make sure data/extracted/ contains the original images.")
        sys.exit(1)

    # Stage 3: Load Ojha model
    model = load_ojha_model()

    # Stage 4+5: Run inference per variant
    print("\n" + "=" * 70)
    print("STAGE 4-5: Running inference on all variants")
    print("=" * 70)

    all_results = {}
    variant_auc = {}
    all_proba_parts = []
    all_true_parts = []
    total_images = 0
    total_inference_time = 0

    for variant in ["original", "whatsapp", "instagram", "screenshot"]:
        t0 = time.time()
        results = run_ojha_inference(model, id_to_path, test_ids, test_labels, variant)
        elapsed_v = time.time() - t0
        n_imgs = len(results["y_proba"])
        total_images += n_imgs
        total_inference_time += elapsed_v

        metrics = compute_metrics(results["y_true"], results["y_proba"])
        metrics["time_sec"] = round(elapsed_v, 1)
        metrics["ms_per_image"] = round(elapsed_v / max(n_imgs, 1) * 1000, 2)
        all_results[variant] = metrics
        variant_auc[variant] = metrics["auc"]
        all_proba_parts.append(results["y_proba"])
        all_true_parts.append(results["y_true"])
        print(f"    {variant}: Acc={metrics['accuracy']}, F1={metrics['f1']}, "
              f"AUC={metrics['auc']}, ECE={metrics['ece']} "
              f"({elapsed_v:.0f}s, {metrics['ms_per_image']:.1f}ms/img)")

    # Combine all variants for main comparison (Table 1)
    print("\n" + "=" * 70)
    print("STAGE 6: Computing combined metrics")
    print("=" * 70)

    combined_proba = np.concatenate(all_proba_parts)
    combined_true = np.concatenate(all_true_parts)
    combined_metrics = compute_metrics(combined_true, combined_proba)

    print(f"\n  COMBINED (all variants, {combined_metrics['n_samples']:,} samples):")
    print(f"    Acc={combined_metrics['accuracy']}, Prec={combined_metrics['precision']}, "
          f"Rec={combined_metrics['recall']}, F1={combined_metrics['f1']}, "
          f"AUC={combined_metrics['auc']}, ECE={combined_metrics['ece']}")

    # Efficiency metrics
    avg_ms = total_inference_time / max(total_images, 1) * 1000
    print(f"\n  Efficiency: {avg_ms:.1f} ms/image (avg across all variants)")
    print(f"  Total: {total_images:,} images in {total_inference_time:.0f}s")

    # Save results
    output = {
        "method": "Ojha et al. (2023) - CLIP ViT-L/14 + linear (UniversalFakeDetect)",
        "model": "fc_weights.pth (frozen CLIP ViT-L/14 + trained linear layer)",
        "combined_metrics": combined_metrics,
        "efficiency": {
            "avg_ms_per_image": round(avg_ms, 2),
            "total_images": total_images,
            "total_time_sec": round(total_inference_time, 1),
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
        },
        "per_variant_metrics": all_results,
        "degradation_summary": {
            "clean": variant_auc.get("original"),
            "whatsapp": variant_auc.get("whatsapp"),
            "instagram": variant_auc.get("instagram"),
            "screenshot": variant_auc.get("screenshot"),
            "avg": round(np.mean(list(variant_auc.values())), 1),
            "clean_to_whatsapp_delta": round(
                variant_auc.get("whatsapp", 0) - variant_auc.get("original", 0), 1
            ),
            "clean_to_instagram_delta": round(
                variant_auc.get("instagram", 0) - variant_auc.get("original", 0), 1
            ),
            "clean_to_screenshot_delta": round(
                variant_auc.get("screenshot", 0) - variant_auc.get("original", 0), 1
            ),
        },
    }

    results_path = OUTPUT_DIR / "ojha_baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Print paper-ready numbers
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"DONE in {elapsed / 60:.1f} minutes")
    print(f"{'=' * 70}")

    print("\n--- FOR TABLE 1 (Main Comparison) ---")
    m = combined_metrics
    print(f"Ojha et al. [2023] & {m['accuracy']} & {m['precision']} & "
          f"{m['recall']} & {m['f1']} & {m['auc']} & {m['ece']}")

    print("\n--- FOR TABLE 3 (Degradation Robustness) ---")
    d = output["degradation_summary"]
    print(f"Ojha et al. [2023] & {d['clean']} & "
          f"{d['whatsapp']}\\,{{\\scriptsize($-${abs(d['clean_to_whatsapp_delta'])})}} & "
          f"{d['instagram']}\\,{{\\scriptsize($-${abs(d['clean_to_instagram_delta'])})}} & "
          f"{d['screenshot']}\\,{{\\scriptsize($-${abs(d['clean_to_screenshot_delta'])})}} & "
          f"{d['avg']}")


if __name__ == "__main__":
    main()
