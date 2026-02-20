"""
Pytest configuration and fixtures for ImageTrust tests.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for test imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


@pytest.fixture(scope="session")
def project_root():
    """Return project root path."""
    return PROJECT_ROOT


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    from PIL import Image

    return Image.new("RGB", (224, 224), color=(128, 128, 128))


@pytest.fixture
def sample_image_path(sample_image, tmp_path):
    """Save sample image to temporary path."""
    path = tmp_path / "test_image.jpg"
    sample_image.save(path, "JPEG", quality=95)
    return path
