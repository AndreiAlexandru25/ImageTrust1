#!/usr/bin/env python3
"""
Download pretrained model weights for ImageTrust.

Usage:
    python scripts/download_models.py --model all
    python scripts/download_models.py --model convnext
    python scripts/download_models.py --model vit
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imagetrust.utils.logging import get_logger, setup_logging
from imagetrust.utils.helpers import ensure_dir

logger = get_logger(__name__)


# Model registry - Add your model URLs here
MODEL_REGISTRY = {
    "convnext_base": {
        "url": None,  # Add HuggingFace or custom URL
        "filename": "convnext_base.pth",
        "description": "ConvNeXt-Base trained on AI detection",
    },
    "vit_base": {
        "url": None,
        "filename": "vit_base.pth",
        "description": "ViT-Base trained on AI detection",
    },
    "efficientnet_b4": {
        "url": None,
        "filename": "efficientnet_b4.pth",
        "description": "EfficientNet-B4 trained on AI detection",
    },
    "ensemble": {
        "url": None,
        "filename": "ensemble_weights.pth",
        "description": "Ensemble model weights",
    },
    "calibration": {
        "url": None,
        "filename": "calibration.pth",
        "description": "Calibration parameters",
    },
}


def download_model(
    model_name: str,
    output_dir: Path,
    force: bool = False,
) -> bool:
    """
    Download a model from the registry.
    
    Args:
        model_name: Name of the model to download
        output_dir: Directory to save the model
        force: Force re-download if exists
        
    Returns:
        True if successful
    """
    if model_name not in MODEL_REGISTRY:
        logger.error(f"Unknown model: {model_name}")
        logger.info(f"Available models: {list(MODEL_REGISTRY.keys())}")
        return False
    
    model_info = MODEL_REGISTRY[model_name]
    output_path = output_dir / model_info["filename"]
    
    if output_path.exists() and not force:
        logger.info(f"Model already exists: {output_path}")
        return True
    
    url = model_info["url"]
    
    if url is None:
        logger.warning(f"No URL configured for {model_name}")
        logger.info("Using pretrained ImageNet weights from timm instead.")
        logger.info("To use custom-trained weights, add URLs to MODEL_REGISTRY")
        return True
    
    logger.info(f"Downloading {model_name} from {url}")
    
    try:
        import urllib.request
        from tqdm import tqdm
        
        # Download with progress bar
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=model_name) as pbar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Successfully downloaded to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {model_name}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def download_all(output_dir: Path, force: bool = False) -> None:
    """Download all models."""
    logger.info(f"Downloading all models to {output_dir}")
    
    success = 0
    failed = 0
    
    for model_name in MODEL_REGISTRY:
        if download_model(model_name, output_dir, force):
            success += 1
        else:
            failed += 1
    
    logger.info(f"Download complete: {success} succeeded, {failed} failed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download pretrained models for ImageTrust"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="all",
        help=f"Model to download. Options: all, {', '.join(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("checkpoints"),
        help="Output directory for models",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download if exists",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models",
    )
    
    args = parser.parse_args()
    
    setup_logging(level="INFO")
    
    if args.list:
        print("\nAvailable models:")
        print("-" * 60)
        for name, info in MODEL_REGISTRY.items():
            print(f"  {name:<20} - {info['description']}")
        print("-" * 60)
        return
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    if args.model == "all":
        download_all(args.output_dir, args.force)
    else:
        download_model(args.model, args.output_dir, args.force)


if __name__ == "__main__":
    main()
