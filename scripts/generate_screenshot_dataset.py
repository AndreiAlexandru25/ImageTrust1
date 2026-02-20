#!/usr/bin/env python
"""
Screenshot/Recapture Dataset Generator for Novel Contribution Evaluation.

Generates synthetic screenshot dataset by:
1. Taking original images
2. Simulating screenshot process (display + capture)
3. Adding typical screenshot artifacts

This enables evaluation of the Screenshot Detection module (Table 7).

Usage:
    python scripts/generate_screenshot_dataset.py --input data/real --output data/screenshots
    python scripts/generate_screenshot_dataset.py --input data/real --output data/screenshots --methods all
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imagetrust.utils.logging import get_logger
from imagetrust.utils.helpers import ensure_dir

logger = get_logger(__name__)

# Common screen resolutions to simulate
SCREEN_CONFIGS = {
    "iphone_14_pro": {
        "resolution": (1290, 2796),
        "ppi": 460,
        "status_bar_height": 54,
        "nav_bar_height": 34,
    },
    "iphone_12": {
        "resolution": (1170, 2532),
        "ppi": 460,
        "status_bar_height": 47,
        "nav_bar_height": 34,
    },
    "samsung_s23": {
        "resolution": (1080, 2340),
        "ppi": 425,
        "status_bar_height": 48,
        "nav_bar_height": 48,
    },
    "desktop_1080p": {
        "resolution": (1920, 1080),
        "ppi": 96,
        "status_bar_height": 0,
        "nav_bar_height": 40,  # Taskbar
    },
    "desktop_1440p": {
        "resolution": (2560, 1440),
        "ppi": 109,
        "status_bar_height": 0,
        "nav_bar_height": 48,
    },
    "macbook_pro": {
        "resolution": (2880, 1800),
        "ppi": 220,
        "status_bar_height": 24,
        "nav_bar_height": 0,
    },
    "ipad_pro": {
        "resolution": (2048, 2732),
        "ppi": 264,
        "status_bar_height": 24,
        "nav_bar_height": 0,
    },
}


def simulate_moire_pattern(image: np.ndarray, intensity: float = 0.05) -> np.ndarray:
    """
    Simulate moiré pattern (screen-door effect from photographing screens).

    This is a key artifact in recaptured images.
    """
    h, w = image.shape[:2]

    # Create interference pattern
    x = np.linspace(0, np.pi * 50, w)
    y = np.linspace(0, np.pi * 50, h)
    xx, yy = np.meshgrid(x, y)

    # Moiré is typically a combination of frequencies
    moire = (
        np.sin(xx * 2 + yy * 2) +
        np.sin(xx * 3 - yy * 3) * 0.5 +
        np.sin(xx * 5 + yy) * 0.3
    )

    moire = (moire - moire.min()) / (moire.max() - moire.min())
    moire = (moire - 0.5) * intensity * 255

    # Apply to all channels
    if len(image.shape) == 3:
        moire = np.stack([moire] * 3, axis=2)

    result = image.astype(np.float32) + moire
    return np.clip(result, 0, 255).astype(np.uint8)


def simulate_color_banding(image: np.ndarray, levels: int = 32) -> np.ndarray:
    """
    Simulate color banding (posterization from limited color depth).

    Common in screenshots due to display color depth limitations.
    """
    # Quantize to fewer levels
    factor = 256 / levels
    quantized = (image / factor).astype(np.uint8) * factor
    return quantized.astype(np.uint8)


def simulate_pixel_grid(image: np.ndarray, grid_intensity: float = 0.03) -> np.ndarray:
    """
    Simulate visible pixel grid from close-up screen capture.
    """
    h, w = image.shape[:2]

    # Create grid pattern
    grid = np.ones((h, w), dtype=np.float32)

    # Horizontal lines (every 2-3 pixels)
    grid[::3, :] *= (1 - grid_intensity)

    # Vertical lines
    grid[:, ::3] *= (1 - grid_intensity)

    if len(image.shape) == 3:
        grid = np.stack([grid] * 3, axis=2)

    return np.clip(image * grid, 0, 255).astype(np.uint8)


def simulate_gamma_shift(image: np.ndarray, gamma: float = 1.1) -> np.ndarray:
    """
    Simulate gamma/brightness shift from screen display.
    """
    normalized = image.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma)
    return (corrected * 255).astype(np.uint8)


def add_status_bar(image: Image.Image, config: Dict[str, Any]) -> Image.Image:
    """
    Add simulated status bar to screenshot.
    """
    status_height = config.get("status_bar_height", 0)
    if status_height == 0:
        return image

    w, h = image.size

    # Create new image with status bar
    new_img = Image.new("RGB", (w, h), color=(0, 0, 0))

    # Draw status bar background (dark gray or black)
    draw = ImageDraw.Draw(new_img)
    draw.rectangle([0, 0, w, status_height], fill=(28, 28, 30))

    # Draw simple status indicators
    try:
        # Battery icon (simple rectangle)
        battery_x = w - 35
        draw.rectangle([battery_x, 12, battery_x + 25, 24], outline=(255, 255, 255), width=1)
        draw.rectangle([battery_x + 25, 15, battery_x + 28, 21], fill=(255, 255, 255))
        draw.rectangle([battery_x + 2, 14, battery_x + 20, 22], fill=(76, 217, 100))

        # WiFi icon (simple bars)
        wifi_x = battery_x - 30
        for i, h_bar in enumerate([4, 7, 10, 13]):
            draw.rectangle([wifi_x + i * 4, 24 - h_bar, wifi_x + i * 4 + 3, 24],
                          fill=(255, 255, 255))

        # Time (center)
        time_str = "9:41"  # Classic Apple time
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), time_str, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text(((w - text_w) // 2, 12), time_str, fill=(255, 255, 255), font=font)
    except Exception:
        pass

    # Paste original image below status bar
    new_img.paste(image.resize((w, h - status_height)), (0, status_height))

    return new_img


def add_navigation_bar(image: Image.Image, config: Dict[str, Any]) -> Image.Image:
    """
    Add simulated navigation/task bar to screenshot.
    """
    nav_height = config.get("nav_bar_height", 0)
    if nav_height == 0:
        return image

    w, h = image.size

    # Create new image with nav bar
    new_img = Image.new("RGB", (w, h), color=(0, 0, 0))

    # Paste original image
    new_img.paste(image.resize((w, h - nav_height)), (0, 0))

    # Draw navigation bar background
    draw = ImageDraw.Draw(new_img)
    draw.rectangle([0, h - nav_height, w, h], fill=(28, 28, 30))

    # Draw simple navigation buttons (iOS style home indicator)
    indicator_width = 134
    indicator_height = 5
    indicator_x = (w - indicator_width) // 2
    indicator_y = h - nav_height // 2 - indicator_height // 2

    draw.rounded_rectangle(
        [indicator_x, indicator_y, indicator_x + indicator_width, indicator_y + indicator_height],
        radius=3,
        fill=(255, 255, 255)
    )

    return new_img


def simulate_screenshot(
    image: Image.Image,
    config: Dict[str, Any],
    methods: List[str],
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Simulate full screenshot process.

    Args:
        image: Original image
        config: Screen configuration
        methods: List of simulation methods to apply

    Returns:
        Tuple of (screenshot_image, metadata)
    """
    metadata = {
        "original_size": image.size,
        "screen_config": config,
        "methods_applied": methods,
    }

    # Resize to screen resolution
    screen_res = config["resolution"]

    # Determine orientation
    if image.width > image.height:
        # Landscape
        target_size = (screen_res[1], screen_res[0]) if screen_res[1] > screen_res[0] else screen_res
    else:
        # Portrait
        target_size = screen_res if screen_res[1] > screen_res[0] else (screen_res[1], screen_res[0])

    # Fit image to screen (with letterboxing)
    aspect_img = image.width / image.height
    aspect_screen = target_size[0] / target_size[1]

    if aspect_img > aspect_screen:
        # Image is wider, fit to width
        new_width = target_size[0]
        new_height = int(new_width / aspect_img)
    else:
        # Image is taller, fit to height
        new_height = target_size[1]
        new_width = int(new_height * aspect_img)

    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create screenshot canvas
    screenshot = Image.new("RGB", target_size, color=(0, 0, 0))

    # Center the image
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    screenshot.paste(resized, (x_offset, y_offset))

    metadata["letterboxed"] = x_offset > 0 or y_offset > 0

    # Convert to numpy for processing
    img_array = np.array(screenshot)

    # Apply selected methods
    if "moire" in methods:
        intensity = random.uniform(0.02, 0.08)
        img_array = simulate_moire_pattern(img_array, intensity)
        metadata["moire_intensity"] = intensity

    if "color_banding" in methods:
        levels = random.choice([24, 32, 48, 64])
        img_array = simulate_color_banding(img_array, levels)
        metadata["color_levels"] = levels

    if "pixel_grid" in methods:
        intensity = random.uniform(0.02, 0.05)
        img_array = simulate_pixel_grid(img_array, intensity)
        metadata["pixel_grid_intensity"] = intensity

    if "gamma" in methods:
        gamma = random.uniform(0.95, 1.15)
        img_array = simulate_gamma_shift(img_array, gamma)
        metadata["gamma"] = gamma

    # Convert back to PIL
    screenshot = Image.fromarray(img_array)

    # Add UI elements
    if "status_bar" in methods:
        screenshot = add_status_bar(screenshot, config)
        metadata["has_status_bar"] = True

    if "nav_bar" in methods:
        screenshot = add_navigation_bar(screenshot, config)
        metadata["has_nav_bar"] = True

    # Apply slight blur (simulates display-camera chain)
    if "blur" in methods:
        blur_radius = random.uniform(0.3, 0.8)
        screenshot = screenshot.filter(ImageFilter.GaussianBlur(blur_radius))
        metadata["blur_radius"] = blur_radius

    metadata["final_size"] = screenshot.size

    return screenshot, metadata


def generate_screenshot_dataset(
    input_dir: Path,
    output_dir: Path,
    methods: List[str],
    num_per_image: int = 3,
    screen_configs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate screenshot dataset from original images.

    Args:
        input_dir: Directory with original images
        output_dir: Output directory for screenshots
        methods: Simulation methods to use
        num_per_image: Number of screenshots per original
        screen_configs: Screen configurations to use

    Returns:
        Dataset metadata
    """
    ensure_dir(output_dir)
    ensure_dir(output_dir / "original")
    ensure_dir(output_dir / "screenshot")

    if screen_configs is None:
        screen_configs = list(SCREEN_CONFIGS.keys())

    # Find all images
    extensions = [".jpg", ".jpeg", ".png", ".webp"]
    images = []
    for ext in extensions:
        images.extend(input_dir.glob(f"*{ext}"))
        images.extend(input_dir.glob(f"*{ext.upper()}"))

    logger.info(f"Found {len(images)} images in {input_dir}")

    dataset_meta = {
        "timestamp": datetime.now().isoformat(),
        "source_dir": str(input_dir),
        "methods": methods,
        "screen_configs": screen_configs,
        "samples": [],
    }

    for i, img_path in enumerate(images):
        logger.info(f"Processing {i+1}/{len(images)}: {img_path.name}")

        try:
            image = Image.open(img_path).convert("RGB")

            # Save original reference
            orig_name = f"orig_{i:05d}{img_path.suffix}"
            image.save(output_dir / "original" / orig_name, quality=95)

            # Generate screenshots
            for j in range(num_per_image):
                # Random configuration
                config_name = random.choice(screen_configs)
                config = SCREEN_CONFIGS[config_name]

                # Random subset of methods (at least 2)
                num_methods = random.randint(2, len(methods))
                selected_methods = random.sample(methods, num_methods)

                # Generate screenshot
                screenshot, meta = simulate_screenshot(image, config, selected_methods)

                # Save screenshot
                screenshot_name = f"screenshot_{i:05d}_{j:02d}.png"
                screenshot.save(output_dir / "screenshot" / screenshot_name)

                # Record metadata
                sample_meta = {
                    "original": orig_name,
                    "screenshot": screenshot_name,
                    "screen_config": config_name,
                    "methods": selected_methods,
                    "details": meta,
                }
                dataset_meta["samples"].append(sample_meta)

        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")

    # Save metadata
    dataset_meta["num_originals"] = len(images)
    dataset_meta["num_screenshots"] = len(dataset_meta["samples"])

    with open(output_dir / "dataset_metadata.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)

    logger.info(f"Generated {dataset_meta['num_screenshots']} screenshots")

    return dataset_meta


def main():
    parser = argparse.ArgumentParser(
        description="Generate screenshot dataset for evaluation"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with original images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/screenshots",
        help="Output directory for screenshots",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated methods or 'all': moire,color_banding,pixel_grid,gamma,blur,status_bar,nav_bar",
    )
    parser.add_argument(
        "--num-per-image",
        type=int,
        default=3,
        help="Number of screenshots per original image",
    )
    parser.add_argument(
        "--screens",
        type=str,
        default=None,
        help="Comma-separated screen configs (default: all)",
    )

    args = parser.parse_args()

    # Parse methods
    if args.methods == "all":
        methods = ["moire", "color_banding", "pixel_grid", "gamma", "blur", "status_bar", "nav_bar"]
    else:
        methods = [m.strip() for m in args.methods.split(",")]

    # Parse screen configs
    screen_configs = None
    if args.screens:
        screen_configs = [s.strip() for s in args.screens.split(",")]

    # Generate dataset
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    logger.info(f"Generating screenshot dataset")
    logger.info(f"Methods: {methods}")
    logger.info(f"Screen configs: {screen_configs or 'all'}")

    metadata = generate_screenshot_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        methods=methods,
        num_per_image=args.num_per_image,
        screen_configs=screen_configs,
    )

    print("\n" + "=" * 60)
    print("SCREENSHOT DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Originals: {metadata['num_originals']}")
    print(f"Screenshots: {metadata['num_screenshots']}")
    print(f"Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
