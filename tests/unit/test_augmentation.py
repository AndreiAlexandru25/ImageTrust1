"""
Unit tests for augmentation module.

Tests the robustness augmentation pipeline including:
- SocialMediaSimulator
- ScreenshotSimulator
- RobustnessAugmentor
"""

import pytest
import numpy as np
from PIL import Image

# Skip if albumentations not available
pytest.importorskip("albumentations")


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    # Create a simple test image (256x256 RGB)
    img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_image_large():
    """Create a larger sample image."""
    img_array = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


class TestSocialMediaSimulator:
    """Tests for SocialMediaSimulator."""

    def test_init(self):
        """Test simulator initialization."""
        from imagetrust.detection.augmentation import SocialMediaSimulator, Platform

        sim = SocialMediaSimulator()
        assert sim.platforms == list(Platform)
        assert sim.apply_double_compression is True

    def test_simulate_instagram(self, sample_image):
        """Test Instagram simulation."""
        from imagetrust.detection.augmentation import SocialMediaSimulator, Platform

        sim = SocialMediaSimulator(platforms=[Platform.INSTAGRAM])
        result, metadata = sim.simulate(sample_image)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert metadata["platform"] == "instagram"
        assert "jpeg_qualities" in metadata

    def test_simulate_whatsapp(self, sample_image):
        """Test WhatsApp simulation (most aggressive compression)."""
        from imagetrust.detection.augmentation import SocialMediaSimulator, Platform

        sim = SocialMediaSimulator(platforms=[Platform.WHATSAPP])
        result, metadata = sim.simulate(sample_image)

        # WhatsApp has aggressive compression
        assert metadata["platform"] == "whatsapp"
        # Check that compression was applied
        assert len(metadata["jpeg_qualities"]) >= 1

    def test_resize_for_platform(self, sample_image_large):
        """Test that large images are resized."""
        from imagetrust.detection.augmentation import SocialMediaSimulator, Platform

        sim = SocialMediaSimulator(platforms=[Platform.INSTAGRAM])
        result, metadata = sim.simulate(sample_image_large)

        # Instagram max is 1080px
        assert max(result.size) <= 1080

    def test_callable(self, sample_image):
        """Test simulator can be called directly."""
        from imagetrust.detection.augmentation import SocialMediaSimulator

        sim = SocialMediaSimulator()
        result = sim(sample_image)

        assert isinstance(result, Image.Image)


class TestScreenshotSimulator:
    """Tests for ScreenshotSimulator."""

    def test_init(self):
        """Test simulator initialization."""
        from imagetrust.detection.augmentation import ScreenshotSimulator

        sim = ScreenshotSimulator()
        assert sim.add_ui_elements is True
        assert sim.add_borders is True

    def test_simulate_windows(self, sample_image):
        """Test Windows screenshot simulation."""
        from imagetrust.detection.augmentation import ScreenshotSimulator, ScreenshotType

        sim = ScreenshotSimulator(screenshot_types=[ScreenshotType.WINDOWS])
        result, metadata = sim.simulate(sample_image)

        assert isinstance(result, Image.Image)
        assert metadata["screenshot_type"] == "windows"
        assert "effects_applied" in metadata

    def test_simulate_mobile(self, sample_image):
        """Test mobile screenshot simulation."""
        from imagetrust.detection.augmentation import ScreenshotSimulator, ScreenshotType

        sim = ScreenshotSimulator(screenshot_types=[ScreenshotType.MOBILE_IOS])
        result, metadata = sim.simulate(sample_image)

        assert metadata["screenshot_type"] == "mobile_ios"

    def test_artifacts_applied(self, sample_image):
        """Test that artifacts are actually applied."""
        from imagetrust.detection.augmentation import ScreenshotSimulator

        sim = ScreenshotSimulator()
        result, metadata = sim.simulate(sample_image)

        # Should have subpixel artifacts and gamma shift at minimum
        assert "subpixel_artifacts" in metadata["effects_applied"]
        assert "gamma_shift" in metadata["effects_applied"]


class TestRobustnessAugmentor:
    """Tests for RobustnessAugmentor."""

    def test_init(self):
        """Test augmentor initialization."""
        from imagetrust.detection.augmentation import RobustnessAugmentor

        aug = RobustnessAugmentor(input_size=224)
        assert aug.input_size == 224

    def test_get_train_transform(self):
        """Test training transform creation."""
        from imagetrust.detection.augmentation import RobustnessAugmentor
        import albumentations as A

        aug = RobustnessAugmentor()
        transform = aug.get_train_transform()

        assert isinstance(transform, A.Compose)

    def test_get_val_transform(self):
        """Test validation transform creation."""
        from imagetrust.detection.augmentation import RobustnessAugmentor
        import albumentations as A

        aug = RobustnessAugmentor()
        transform = aug.get_val_transform()

        assert isinstance(transform, A.Compose)

    def test_preprocess_train(self, sample_image):
        """Test training preprocessing."""
        from imagetrust.detection.augmentation import RobustnessAugmentor
        import torch

        aug = RobustnessAugmentor(input_size=224)
        tensor, metadata = aug.apply_pil_augmentation(sample_image, mode="train")

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)

    def test_preprocess_val(self, sample_image):
        """Test validation preprocessing."""
        from imagetrust.detection.augmentation import RobustnessAugmentor
        import torch

        aug = RobustnessAugmentor(input_size=224)
        tensor, metadata = aug.apply_pil_augmentation(sample_image, mode="val")

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)

    def test_callable(self, sample_image):
        """Test augmentor can be called directly."""
        from imagetrust.detection.augmentation import RobustnessAugmentor
        import torch

        aug = RobustnessAugmentor(input_size=224)
        tensor = aug(sample_image, mode="val")

        assert isinstance(tensor, torch.Tensor)


class TestCreateRobustnessTransform:
    """Tests for factory function."""

    def test_create_transform(self):
        """Test transform factory function."""
        from imagetrust.detection.augmentation import create_robustness_dataset_transform

        transform = create_robustness_dataset_transform(input_size=224, mode="train")
        assert callable(transform)

    def test_transform_output(self, sample_image):
        """Test transform output shape."""
        from imagetrust.detection.augmentation import create_robustness_dataset_transform
        import torch

        transform = create_robustness_dataset_transform(input_size=224, mode="val")
        tensor = transform(sample_image)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
