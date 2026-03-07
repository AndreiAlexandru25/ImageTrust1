"""
Generate example images figure for thesis/paper.

Produces fig10_example_images.png / .pdf:
  2x3 grid showing real vs AI-generated images from the dataset.
  Top row: Real photographs (COCO, FFHQ, Deepfake-Real)
  Bottom row: AI-generated (CIFAKE-SD, SFHQ, Deepfake-Fake)
"""

import pathlib
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "extracted"
OUT = ROOT / "outputs" / "phase3" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def pick_image(folder, n=1):
    """Pick n random images from folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [f for f in pathlib.Path(folder).iterdir()
            if f.suffix.lower() in exts]
    if not imgs:
        return None
    return random.sample(imgs, min(n, len(imgs)))


def load_and_crop(path, size=224):
    """Load image and center-crop to square."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((size, size), Image.LANCZOS)
    return img


def main():
    # ── Select images ────────────────────────────────────────────────────
    sources = {
        # Top row: REAL
        "Real (COCO)": DATA / "COCO 2017 dataset" / "coco2017",
        "Real (FFHQ)": DATA / "Flickr-Faces-HQ Dataset(FFHQ)",
        "Real (Deepfake-Real)": DATA / "deepfake and real images" / "Dataset" / "Train" / "Real",
        # Bottom row: AI-GENERATED
        "AI-Gen. (CIFAKE-SD)": (
            DATA / "CIFAKE Real and AI-Generated Synthetic Images" / "train" / "FAKE"
        ),
        "AI-Gen. (SFHQ)": (
            DATA / "Synthetic Faces High Quality SFHQ part 1" / "a tiny sample (30 images)"
        ),
        "AI-Gen. (Deepfake)": (
            DATA / "deepfake and real images" / "Dataset" / "Train" / "Fake"
        ),
    }

    # Try to find COCO images in nested directory
    coco_path = sources["Real (COCO)"]
    if coco_path.is_dir():
        # COCO might have subdirectories
        sub = list(coco_path.iterdir())
        img_exts = {".jpg", ".jpeg", ".png"}
        has_imgs = any(f.suffix.lower() in img_exts for f in sub if f.is_file())
        if not has_imgs:
            # Check subdirectories
            for s in sub:
                if s.is_dir():
                    inner = list(s.iterdir())
                    if any(f.suffix.lower() in img_exts for f in inner if f.is_file()):
                        sources["Real (COCO)"] = s
                        break

    images = {}
    for label, folder in sources.items():
        folder = pathlib.Path(folder)
        if not folder.exists():
            print(f"WARNING: {folder} not found, skipping {label}")
            continue
        picks = pick_image(folder)
        if picks:
            images[label] = picks[0]
            print(f"  {label}: {picks[0].name}")
        else:
            print(f"WARNING: No images found in {folder}")

    if len(images) < 4:
        print("ERROR: Not enough images found. Need at least 4.")
        return

    # ── Build figure ─────────────────────────────────────────────────────
    n_cols = 3
    n_rows = 2

    # Separate real and AI labels
    real_labels = [k for k in images if k.startswith("Real")]
    ai_labels = [k for k in images if k.startswith("AI")]

    # Ensure we have enough for the grid
    row_labels = [real_labels[:n_cols], ai_labels[:n_cols]]
    all_labels = row_labels[0] + row_labels[1]

    actual_cols = max(len(row_labels[0]), len(row_labels[1]))
    actual_rows = 2

    fig, axes = plt.subplots(actual_rows, actual_cols, figsize=(3.3 * actual_cols, 3.8 * actual_rows))
    fig.suptitle("Example Images from the Training Dataset", fontsize=14, y=1.02)

    # Row headers
    row_titles = [
        r"$\mathbf{Authentic\ (Real)\ Photographs}$",
        r"$\mathbf{AI\text{-}Generated\ Images}$",
    ]

    for row_idx, (labels, title) in enumerate(zip(row_labels, row_titles)):
        for col_idx in range(actual_cols):
            ax = axes[row_idx, col_idx] if actual_rows > 1 else axes[col_idx]
            if col_idx < len(labels):
                label = labels[col_idx]
                img = load_and_crop(images[label], size=256)
                ax.imshow(img)
                # Title below image
                short_label = label.replace("Real ", "").replace("AI-Gen. ", "")
                ax.set_xlabel(short_label, fontsize=10)
            else:
                ax.axis("off")
                continue

            ax.set_xticks([])
            ax.set_yticks([])

            # Color border: green for real, red for AI
            color = "#4CAF50" if row_idx == 0 else "#F44336"
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)

        # Row label on the left
        if actual_cols > 0:
            left_ax = axes[row_idx, 0] if actual_rows > 1 else axes[0]
            left_ax.set_ylabel(title, fontsize=11, labelpad=10)

    plt.tight_layout()

    for ext in ("png", "pdf"):
        path = OUT / f"fig10_example_images.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
