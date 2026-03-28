"""
Robustness Augmentation Pipeline for AI Detection.

Implements domain-specific augmentations to reduce False Positives on:
- Social media images (double JPEG compression, chroma subsampling)
- Screenshots (UI overlays, anti-aliased text, window borders)
- Printscreens (sub-pixel rendering, recapture noise)

This module is critical for international publication-level robustness.
"""

import io
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class Platform(Enum):
    """Social media platforms with specific compression profiles."""
    INSTAGRAM = "instagram"
    WHATSAPP = "whatsapp"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    TELEGRAM = "telegram"
    GENERIC = "generic"


class ScreenshotType(Enum):
    """Screenshot/printscreen types."""
    WINDOWS = "windows"
    MACOS = "macos"
    MOBILE_IOS = "mobile_ios"
    MOBILE_ANDROID = "mobile_android"
    BROWSER = "browser"
    GENERIC = "generic"


@dataclass
class PlatformProfile:
    """Compression profile for a social media platform."""
    name: str
    jpeg_quality_range: Tuple[int, int]
    max_dimension: int
    chroma_subsampling: str  # "4:2:0", "4:2:2", "4:4:4"
    apply_resize: bool
    additional_blur: float  # Gaussian blur sigma
    noise_level: float  # Gaussian noise stddev


# Platform-specific compression profiles (empirically derived)
PLATFORM_PROFILES: Dict[Platform, PlatformProfile] = {
    Platform.INSTAGRAM: PlatformProfile(
        name="Instagram",
        jpeg_quality_range=(65, 75),
        max_dimension=1080,
        chroma_subsampling="4:2:0",
        apply_resize=True,
        additional_blur=0.3,
        noise_level=0.5,
    ),
    Platform.WHATSAPP: PlatformProfile(
        name="WhatsApp",
        jpeg_quality_range=(45, 60),
        max_dimension=1600,
        chroma_subsampling="4:2:0",
        apply_resize=True,
        additional_blur=0.5,
        noise_level=1.0,
    ),
    Platform.TWITTER: PlatformProfile(
        name="Twitter/X",
        jpeg_quality_range=(80, 90),
        max_dimension=4096,
        chroma_subsampling="4:2:0",
        apply_resize=False,
        additional_blur=0.1,
        noise_level=0.2,
    ),
    Platform.FACEBOOK: PlatformProfile(
        name="Facebook",
        jpeg_quality_range=(70, 85),
        max_dimension=2048,
        chroma_subsampling="4:2:0",
        apply_resize=True,
        additional_blur=0.2,
        noise_level=0.3,
    ),
    Platform.TELEGRAM: PlatformProfile(
        name="Telegram",
        jpeg_quality_range=(75, 85),
        max_dimension=2560,
        chroma_subsampling="4:2:0",
        apply_resize=True,
        additional_blur=0.1,
        noise_level=0.2,
    ),
    Platform.GENERIC: PlatformProfile(
        name="Generic",
        jpeg_quality_range=(50, 95),
        max_dimension=2048,
        chroma_subsampling="4:2:0",
        apply_resize=True,
        additional_blur=0.3,
        noise_level=0.5,
    ),
}


class SocialMediaSimulator:
    """
    Simulate social media platform compression artifacts.

    Implements realistic degradation patterns observed in images shared
    through various social media platforms:
    - Double/triple JPEG recompression
    - Chroma subsampling (4:2:0)
    - Platform-specific resizing
    - Additional blur and noise from processing pipelines
    """

    def __init__(
        self,
        platforms: Optional[List[Platform]] = None,
        apply_double_compression: bool = True,
        compression_rounds: int = 2,
    ):
        """
        Initialize social media simulator.

        Args:
            platforms: List of platforms to simulate. None = all platforms.
            apply_double_compression: Apply JPEG compression multiple times.
            compression_rounds: Number of compression rounds (1-3).
        """
        self.platforms = platforms or list(Platform)
        self.apply_double_compression = apply_double_compression
        self.compression_rounds = min(max(compression_rounds, 1), 3)

    def _apply_jpeg_compression(
        self,
        image: Image.Image,
        quality: int,
        subsampling: str = "4:2:0",
    ) -> Image.Image:
        """Apply JPEG compression with specific quality and subsampling."""
        # Map subsampling string to PIL parameter
        subsampling_map = {
            "4:4:4": 0,  # No subsampling
            "4:2:2": 1,  # Horizontal subsampling
            "4:2:0": 2,  # Full subsampling (most aggressive)
        }
        subsampling_value = subsampling_map.get(subsampling, 2)

        # Compress to JPEG in memory
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=subsampling_value,
        )
        buffer.seek(0)

        return Image.open(buffer).convert("RGB")

    def _resize_for_platform(
        self,
        image: Image.Image,
        profile: PlatformProfile,
    ) -> Image.Image:
        """Resize image according to platform constraints."""
        if not profile.apply_resize:
            return image

        width, height = image.size
        max_dim = profile.max_dimension

        if max(width, height) <= max_dim:
            return image

        # Scale down maintaining aspect ratio
        scale = max_dim / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _add_processing_artifacts(
        self,
        image: Image.Image,
        profile: PlatformProfile,
    ) -> Image.Image:
        """Add blur and noise typical of platform processing pipelines."""
        img_array = np.array(image, dtype=np.float32)

        # Add Gaussian noise
        if profile.noise_level > 0:
            noise = np.random.normal(0, profile.noise_level, img_array.shape)
            img_array = img_array + noise
            img_array = np.clip(img_array, 0, 255)

        result = Image.fromarray(img_array.astype(np.uint8))

        # Add slight blur
        if profile.additional_blur > 0:
            result = result.filter(
                ImageFilter.GaussianBlur(radius=profile.additional_blur)
            )

        return result

    def simulate(
        self,
        image: Image.Image,
        platform: Optional[Platform] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Simulate social media platform compression.

        Args:
            image: Input PIL Image.
            platform: Specific platform to simulate. None = random selection.

        Returns:
            Tuple of (degraded_image, metadata_dict).
        """
        if platform is None:
            platform = random.choice(self.platforms)

        profile = PLATFORM_PROFILES[platform]

        # Start with original image
        result = image.convert("RGB")

        # Resize for platform
        result = self._resize_for_platform(result, profile)

        # Apply compression rounds
        qualities_used = []
        for _ in range(self.compression_rounds if self.apply_double_compression else 1):
            quality = random.randint(*profile.jpeg_quality_range)
            qualities_used.append(quality)
            result = self._apply_jpeg_compression(
                result,
                quality=quality,
                subsampling=profile.chroma_subsampling,
            )

        # Add processing artifacts
        result = self._add_processing_artifacts(result, profile)

        metadata = {
            "platform": platform.value,
            "profile_name": profile.name,
            "jpeg_qualities": qualities_used,
            "compression_rounds": len(qualities_used),
            "chroma_subsampling": profile.chroma_subsampling,
            "original_size": image.size,
            "final_size": result.size,
        }

        return result, metadata

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply random platform simulation."""
        result, _ = self.simulate(image)
        return result


class ScreenshotSimulator:
    """
    Simulate screenshot/printscreen artifacts.

    Generates realistic screenshot-like degradations:
    - UI overlays (buttons, icons, status bars)
    - Anti-aliased text rendering
    - Window borders and shadows
    - Sub-pixel rendering artifacts
    - Screen capture gamma shifts
    """

    # Common UI element colors
    UI_COLORS = {
        "light_bg": [(248, 248, 248), (255, 255, 255), (240, 240, 240)],
        "dark_bg": [(30, 30, 30), (45, 45, 45), (60, 60, 60)],
        "accent": [(0, 122, 255), (88, 86, 214), (255, 59, 48)],
        "text": [(0, 0, 0), (60, 60, 60), (255, 255, 255)],
    }

    def __init__(
        self,
        screenshot_types: Optional[List[ScreenshotType]] = None,
        add_ui_elements: bool = True,
        add_borders: bool = True,
        add_text_overlays: bool = True,
        ui_coverage: float = 0.15,  # Max 15% of image covered by UI
    ):
        """
        Initialize screenshot simulator.

        Args:
            screenshot_types: Types to simulate. None = all types.
            add_ui_elements: Add UI elements (buttons, icons).
            add_borders: Add window borders.
            add_text_overlays: Add text overlays.
            ui_coverage: Maximum fraction of image covered by UI elements.
        """
        self.screenshot_types = screenshot_types or list(ScreenshotType)
        self.add_ui_elements = add_ui_elements
        self.add_borders = add_borders
        self.add_text_overlays = add_text_overlays
        self.ui_coverage = ui_coverage

    def _add_status_bar(
        self,
        image: Image.Image,
        position: str = "top",
        dark_mode: bool = False,
    ) -> Image.Image:
        """Add a status bar (mobile) or menu bar (desktop)."""
        width, height = image.size
        bar_height = max(24, int(height * 0.04))

        # Create status bar
        bg_color = random.choice(
            self.UI_COLORS["dark_bg" if dark_mode else "light_bg"]
        )
        text_color = random.choice(
            self.UI_COLORS["text"][:1] if not dark_mode else self.UI_COLORS["text"][2:]
        )

        # Create new image with status bar
        new_height = height + bar_height
        result = Image.new("RGB", (width, new_height), bg_color)

        if position == "top":
            result.paste(image, (0, bar_height))
        else:
            result.paste(image, (0, 0))

        # Draw status bar elements
        draw = ImageDraw.Draw(result)

        # Time
        y_pos = 6 if position == "top" else height + 6
        draw.text((10, y_pos), "9:41", fill=text_color)

        # Battery icon (simple rectangle)
        battery_x = width - 35
        draw.rectangle(
            [battery_x, y_pos, battery_x + 25, y_pos + 12],
            outline=text_color,
            width=1,
        )
        draw.rectangle(
            [battery_x + 2, y_pos + 2, battery_x + 20, y_pos + 10],
            fill=text_color,
        )

        return result

    def _add_window_border(
        self,
        image: Image.Image,
        style: str = "modern",
    ) -> Image.Image:
        """Add window border and title bar."""
        width, height = image.size
        border_width = random.randint(1, 3)
        title_bar_height = random.randint(28, 38)

        # Window chrome colors
        if style == "modern":
            bg_color = (240, 240, 240)
            border_color = (200, 200, 200)
            button_colors = [(255, 95, 87), (255, 189, 46), (39, 201, 63)]
        else:
            bg_color = (220, 220, 220)
            border_color = (180, 180, 180)
            button_colors = [(255, 95, 87), (255, 189, 46), (39, 201, 63)]

        # Create window with border
        new_width = width + 2 * border_width
        new_height = height + title_bar_height + 2 * border_width

        result = Image.new("RGB", (new_width, new_height), border_color)

        # Title bar
        draw = ImageDraw.Draw(result)
        draw.rectangle(
            [border_width, border_width, new_width - border_width, border_width + title_bar_height],
            fill=bg_color,
        )

        # Window buttons (macOS style)
        button_y = border_width + title_bar_height // 2
        for i, color in enumerate(button_colors):
            button_x = border_width + 12 + i * 20
            draw.ellipse(
                [button_x - 6, button_y - 6, button_x + 6, button_y + 6],
                fill=color,
            )

        # Paste original image
        result.paste(image, (border_width, border_width + title_bar_height))

        return result

    def _add_text_overlay(
        self,
        image: Image.Image,
        text: Optional[str] = None,
    ) -> Image.Image:
        """Add anti-aliased text overlay (like watermarks or UI text)."""
        result = image.copy()
        draw = ImageDraw.Draw(result)

        width, height = image.size

        # Generate random text if not provided
        if text is None:
            texts = [
                "Screenshot", "Image", "Photo", "View",
                "www.", ".com", "Copy", "Save", "Share",
            ]
            text = random.choice(texts)

        # Random position (avoiding center to not obscure main content)
        positions = [
            (10, 10),  # Top-left
            (width - 80, 10),  # Top-right
            (10, height - 30),  # Bottom-left
            (width - 80, height - 30),  # Bottom-right
        ]
        pos = random.choice(positions)

        # Semi-transparent text
        text_color = (128, 128, 128)
        draw.text(pos, text, fill=text_color)

        return result

    def _apply_subpixel_artifacts(
        self,
        image: Image.Image,
        intensity: float = 0.3,
    ) -> Image.Image:
        """Simulate sub-pixel rendering artifacts from screen capture."""
        img_array = np.array(image, dtype=np.float32)

        # Slight color channel shifts (simulating RGB sub-pixel alignment)
        shift = int(intensity * 2)
        if shift > 0:
            # Shift red channel slightly right, blue slightly left
            img_array[:, shift:, 0] = img_array[:, :-shift, 0]
            img_array[:, :-shift, 2] = img_array[:, shift:, 2]

        # Add subtle noise pattern (screen capture noise)
        noise = np.random.normal(0, intensity * 2, img_array.shape)
        img_array = img_array + noise
        img_array = np.clip(img_array, 0, 255)

        return Image.fromarray(img_array.astype(np.uint8))

    def _apply_gamma_shift(
        self,
        image: Image.Image,
        gamma_range: Tuple[float, float] = (0.95, 1.05),
    ) -> Image.Image:
        """Apply gamma correction shift (common in screen captures)."""
        gamma = random.uniform(*gamma_range)

        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.power(img_array, gamma)
        img_array = (img_array * 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def simulate(
        self,
        image: Image.Image,
        screenshot_type: Optional[ScreenshotType] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Simulate screenshot artifacts.

        Args:
            image: Input PIL Image.
            screenshot_type: Type of screenshot. None = random selection.

        Returns:
            Tuple of (screenshot_image, metadata_dict).
        """
        if screenshot_type is None:
            screenshot_type = random.choice(self.screenshot_types)

        result = image.convert("RGB")
        applied_effects = []

        # Apply effects based on type and settings
        if screenshot_type in [ScreenshotType.MOBILE_IOS, ScreenshotType.MOBILE_ANDROID]:
            if self.add_ui_elements and random.random() < 0.7:
                result = self._add_status_bar(
                    result,
                    position="top",
                    dark_mode=random.random() < 0.5,
                )
                applied_effects.append("status_bar")

        if screenshot_type in [ScreenshotType.WINDOWS, ScreenshotType.MACOS, ScreenshotType.BROWSER]:
            if self.add_borders and random.random() < 0.6:
                result = self._add_window_border(result)
                applied_effects.append("window_border")

        if self.add_text_overlays and random.random() < 0.3:
            result = self._add_text_overlay(result)
            applied_effects.append("text_overlay")

        # Always apply sub-pixel artifacts for screenshots
        result = self._apply_subpixel_artifacts(
            result,
            intensity=random.uniform(0.1, 0.4),
        )
        applied_effects.append("subpixel_artifacts")

        # Gamma shift (screen to sRGB conversion)
        result = self._apply_gamma_shift(result)
        applied_effects.append("gamma_shift")

        metadata = {
            "screenshot_type": screenshot_type.value,
            "effects_applied": applied_effects,
            "original_size": image.size,
            "final_size": result.size,
        }

        return result, metadata

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply random screenshot simulation."""
        result, _ = self.simulate(image)
        return result


class RobustnessAugmentor:
    """
    Main Albumentations-based augmentation pipeline for robust AI detection training.

    Combines:
    - Standard augmentations (flip, rotate, color jitter)
    - Social media simulation (JPEG compression, chroma subsampling)
    - Screenshot simulation (UI overlays, sub-pixel artifacts)
    - Adversarial-style perturbations

    This pipeline is designed to train models that are robust to real-world
    image degradations that commonly cause False Positives.
    """

    def __init__(
        self,
        input_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        social_media_prob: float = 0.3,
        screenshot_prob: float = 0.2,
        heavy_compression_prob: float = 0.15,
    ):
        """
        Initialize robustness augmentor.

        Args:
            input_size: Target image size.
            mean: Normalization mean (ImageNet defaults).
            std: Normalization std (ImageNet defaults).
            social_media_prob: Probability of applying social media simulation.
            screenshot_prob: Probability of applying screenshot simulation.
            heavy_compression_prob: Probability of heavy JPEG compression.
        """
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.social_media_prob = social_media_prob
        self.screenshot_prob = screenshot_prob
        self.heavy_compression_prob = heavy_compression_prob

        # Simulators
        self.social_media_sim = SocialMediaSimulator()
        self.screenshot_sim = ScreenshotSimulator()

        # Check Albumentations availability
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError(
                "Albumentations is required for RobustnessAugmentor. "
                "Install with: pip install albumentations"
            )

    def get_train_transform(self) -> A.Compose:
        """
        Get training augmentation pipeline.

        Includes aggressive augmentations for robustness training.
        """
        return A.Compose([
            # Spatial transforms
            A.RandomResizedCrop(
                size=(self.input_size, self.input_size),
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                border_mode=0,
                p=0.3,
            ),

            # Color transforms
            A.OneOf([
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                    p=1.0,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0,
                ),
            ], p=0.5),

            # Compression artifacts (crucial for robustness)
            A.OneOf([
                A.ImageCompression(
                    quality_lower=30,
                    quality_upper=60,
                    p=1.0,
                ),
                A.ImageCompression(
                    quality_lower=60,
                    quality_upper=85,
                    p=1.0,
                ),
                A.ImageCompression(
                    quality_lower=85,
                    quality_upper=100,
                    p=1.0,
                ),
            ], p=self.heavy_compression_prob + 0.3),

            # Blur (simulates various processing)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.2),

            # Noise (sensor noise, compression artifacts)
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
            ], p=0.25),

            # Downscale-upscale (simulates resizing pipelines)
            A.OneOf([
                A.Downscale(
                    scale_min=0.5,
                    scale_max=0.75,
                    interpolation=0,  # INTER_NEAREST
                    p=1.0,
                ),
                A.Downscale(
                    scale_min=0.5,
                    scale_max=0.75,
                    interpolation=1,  # INTER_LINEAR
                    p=1.0,
                ),
            ], p=0.2),

            # Normalize and convert to tensor
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

    def get_val_transform(self) -> A.Compose:
        """
        Get validation/test augmentation pipeline.

        Minimal augmentations for fair evaluation.
        """
        return A.Compose([
            A.Resize(height=self.input_size, width=self.input_size),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

    def get_social_media_transform(self) -> A.Compose:
        """
        Get social media-focused augmentation pipeline.

        Heavy emphasis on compression and resizing artifacts.
        """
        return A.Compose([
            A.Resize(height=self.input_size, width=self.input_size),

            # Heavy JPEG compression (double compression simulation)
            A.ImageCompression(
                quality_lower=40,
                quality_upper=75,
                p=0.8,
            ),

            # Additional compression round
            A.ImageCompression(
                quality_lower=50,
                quality_upper=85,
                p=0.5,
            ),

            # Color space shifts (chroma subsampling artifacts)
            A.ToSepia(p=0.05),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),

            # Slight blur from processing
            A.GaussianBlur(blur_limit=(1, 3), p=0.4),

            # Noise from multiple encoding passes
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),

            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

    def get_screenshot_transform(self) -> A.Compose:
        """
        Get screenshot-focused augmentation pipeline.

        Emphasis on rendering artifacts and UI element patterns.
        """
        return A.Compose([
            A.Resize(height=self.input_size, width=self.input_size),

            # Sub-pixel rendering artifacts (channel shifts)
            A.ChannelShuffle(p=0.1),
            A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.4),

            # Gamma shifts from display capture
            A.RandomGamma(gamma_limit=(90, 110), p=0.5),

            # Sharp edges from UI elements
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=0.3),

            # Screen capture compression (usually PNG or high-quality JPEG)
            A.ImageCompression(
                quality_lower=85,
                quality_upper=100,
                p=0.4,
            ),

            # Slight posterization (color quantization)
            A.Posterize(num_bits=6, p=0.15),

            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

    def get_combined_transform(self) -> A.Compose:
        """
        Get combined robustness training pipeline.

        Randomly applies standard, social media, or screenshot augmentations.
        """
        return A.Compose([
            A.RandomResizedCrop(
                size=(self.input_size, self.input_size),
                scale=(0.8, 1.0),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),

            # Random augmentation type selection
            A.OneOf([
                # Standard augmentation
                A.Compose([
                    A.ColorJitter(
                        brightness=0.15,
                        contrast=0.15,
                        saturation=0.15,
                        hue=0.03,
                        p=0.5,
                    ),
                    A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
                ]),

                # Social media simulation
                A.Compose([
                    A.ImageCompression(quality_lower=40, quality_upper=70, p=0.9),
                    A.ImageCompression(quality_lower=50, quality_upper=80, p=0.5),
                    A.GaussianBlur(blur_limit=(1, 3), p=0.3),
                    A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
                ]),

                # Screenshot simulation
                A.Compose([
                    A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.5),
                    A.RandomGamma(gamma_limit=(95, 105), p=0.4),
                    A.Sharpen(alpha=(0.1, 0.2), p=0.3),
                    A.ImageCompression(quality_lower=90, quality_upper=100, p=0.3),
                ]),
            ], p=0.7),

            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

    def apply_pil_augmentation(
        self,
        image: Image.Image,
        mode: str = "train",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply PIL-based augmentation with simulators.

        Args:
            image: Input PIL Image.
            mode: Augmentation mode ("train", "social_media", "screenshot", "combined").

        Returns:
            Tuple of (augmented_tensor, metadata).
        """
        metadata = {"mode": mode, "augmentations": []}

        # Apply simulator-based augmentations (PIL domain)
        if mode == "train" or mode == "combined":
            if random.random() < self.social_media_prob:
                image, sim_meta = self.social_media_sim.simulate(image)
                metadata["augmentations"].append(("social_media", sim_meta))
            elif random.random() < self.screenshot_prob:
                image, sim_meta = self.screenshot_sim.simulate(image)
                metadata["augmentations"].append(("screenshot", sim_meta))
        elif mode == "social_media":
            image, sim_meta = self.social_media_sim.simulate(image)
            metadata["augmentations"].append(("social_media", sim_meta))
        elif mode == "screenshot":
            image, sim_meta = self.screenshot_sim.simulate(image)
            metadata["augmentations"].append(("screenshot", sim_meta))

        # Convert to numpy for Albumentations
        img_array = np.array(image)

        # Get appropriate transform
        if mode == "train":
            transform = self.get_train_transform()
        elif mode == "social_media":
            transform = self.get_social_media_transform()
        elif mode == "screenshot":
            transform = self.get_screenshot_transform()
        elif mode == "combined":
            transform = self.get_combined_transform()
        else:
            transform = self.get_val_transform()

        # Apply Albumentations transform
        augmented = transform(image=img_array)
        tensor = augmented["image"]

        return tensor, metadata

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        mode: str = "train",
    ) -> np.ndarray:
        """Apply augmentation and return tensor."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        tensor, _ = self.apply_pil_augmentation(image, mode)
        return tensor


def create_robustness_dataset_transform(
    input_size: int = 224,
    mode: str = "train",
    social_media_prob: float = 0.3,
    screenshot_prob: float = 0.2,
) -> Callable:
    """
    Factory function to create transform for PyTorch Dataset.

    Args:
        input_size: Target image size.
        mode: Transform mode ("train", "val", "social_media", "screenshot", "combined").
        social_media_prob: Probability of social media simulation.
        screenshot_prob: Probability of screenshot simulation.

    Returns:
        Callable transform function.
    """
    augmentor = RobustnessAugmentor(
        input_size=input_size,
        social_media_prob=social_media_prob,
        screenshot_prob=screenshot_prob,
    )

    def transform(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        return augmentor(image, mode=mode)

    return transform
