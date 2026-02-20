"""
Fix mislabeled shards from Phase 1.

The infer_label_from_path() function incorrectly labeled ALL images as AI (label=1)
because filenames like 'cifake_real_REAL_0000.jpg' contain 'fake' in 'cifake'.

This script fixes labels in-place based on image_id patterns:

REAL (label=0):
  - cifake_real_REAL_*       -> CIFAKE real photos
  - coco_*                   -> COCO 2017 (real photos)
  - ffhq_*                   -> Flickr-Faces-HQ (real photos)
  - deepfake_deepfake and real images_real_*  -> real subset
  - other_real_*             -> other real images

AI/FAKE (label=1):
  - cifake_sd_FAKE_*         -> CIFAKE Stable Diffusion
  - deepfake_faces_deepfake_* -> deepfake faces
  - deepfake_deepfake and real images_fake_* -> fake subset
  - sfhq_*                   -> Synthetic Faces HQ
  - hard_fakes_fake_*        -> hard fakes
"""

import json
import numpy as np
from pathlib import Path
import time
import sys


def infer_label_from_image_id(image_id: str) -> int:
    """Correctly infer label from image_id pattern. 0=real, 1=AI."""
    s = image_id.lower()

    # Definite REAL datasets (label=0)
    if s.startswith("cifake_real_"):
        return 0
    if s.startswith("coco_"):
        return 0
    if s.startswith("ffhq_"):
        return 0
    if s.startswith("other_real_"):
        return 0
    if "real images_real" in s or "_real_real" in s:
        return 0
    if s.startswith("deepfake_deepfake and real images_real"):
        return 0

    # Definite FAKE/AI datasets (label=1)
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

    # Default: if truly unknown, mark as real (conservative)
    return 0


def main():
    embeddings_dir = Path("data/phase1/embeddings")
    index_path = embeddings_dir / "embedding_index.json"

    if not index_path.exists():
        print("ERROR: embedding_index.json not found")
        sys.exit(1)

    with open(index_path) as f:
        index = json.load(f)

    total_shards = len(index["shard_files"])
    print(f"Fixing labels in {total_shards} shards...")

    total_fixed = 0
    total_samples = 0
    label_counts = {0: 0, 1: 0}
    t0 = time.time()

    for i, shard_file in enumerate(index["shard_files"]):
        shard_path = embeddings_dir / shard_file
        if not shard_path.exists():
            continue

        shard = np.load(shard_path, allow_pickle=True)
        image_ids = shard["image_ids"]
        old_labels = shard["labels"]
        new_labels = np.array(
            [infer_label_from_image_id(str(iid)) for iid in image_ids],
            dtype=np.int32,
        )

        changed = int(np.sum(old_labels != new_labels))
        total_fixed += changed
        total_samples += len(new_labels)
        for lbl in [0, 1]:
            label_counts[lbl] += int((new_labels == lbl).sum())

        # Rewrite shard with corrected labels
        data = {key: shard[key] for key in shard.files}
        data["labels"] = new_labels
        shard.close()

        np.savez_compressed(shard_path, **data)

        if (i + 1) % 50 == 0 or (i + 1) == total_shards:
            elapsed = time.time() - t0
            print(
                f"  {i+1}/{total_shards} shards | "
                f"{total_fixed:,} labels fixed | "
                f"real={label_counts[0]:,} ai={label_counts[1]:,} | "
                f"{elapsed:.0f}s"
            )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Total samples: {total_samples:,}")
    print(f"Labels fixed:  {total_fixed:,}")
    print(f"Final: real={label_counts[0]:,}, ai={label_counts[1]:,}")
    ratio = label_counts[0] / max(label_counts[1], 1)
    print(f"Ratio real/ai: {ratio:.2f}")


if __name__ == "__main__":
    main()
