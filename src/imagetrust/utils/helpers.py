"""
General utility and helper functions for ImageTrust.
"""

import hashlib
import time
import uuid
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

from PIL import Image

from imagetrust.core.exceptions import InvalidImageError


def generate_id(prefix: str = "analysis") -> str:
    """
    Generate a unique ID with optional prefix.
    
    Args:
        prefix: Prefix for the ID.
        
    Returns:
        A unique string ID.
    """
    timestamp = int(time.time() * 1000)
    unique = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{unique}"


def ensure_dir(path: Union[Path, str]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Directory path.
        
    Returns:
        The Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def calculate_file_hash(
    file_path: Optional[Union[Path, str]] = None,
    data: Optional[bytes] = None,
    algorithm: str = "sha256",
) -> str:
    """
    Calculate the hash of a file or bytes.
    
    Args:
        file_path: Path to the file.
        data: Raw bytes (alternative to file_path).
        algorithm: Hash algorithm (sha256, md5, etc.).
        
    Returns:
        Hex digest of the hash.
    """
    hasher = hashlib.new(algorithm)
    
    if data is not None:
        hasher.update(data)
    elif file_path is not None:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
    else:
        raise ValueError("Either file_path or data must be provided")
    
    return hasher.hexdigest()


@contextmanager
def timer() -> Generator[Dict[str, float], None, None]:
    """
    Context manager to time code execution.
    
    Yields:
        Dictionary with 'elapsed_s' and 'elapsed_ms'.
        
    Example:
        with timer() as t:
            do_something()
        print(f"Took {t['elapsed_ms']:.2f} ms")
    """
    result = {"elapsed_s": 0.0, "elapsed_ms": 0.0}
    start_time = time.perf_counter()
    try:
        yield result
    finally:
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        result["elapsed_s"] = elapsed
        result["elapsed_ms"] = elapsed * 1000


def load_image(
    source: Union[Path, str, bytes, BytesIO, Image.Image],
    convert_rgb: bool = True,
) -> Image.Image:
    """
    Load an image from various sources.
    
    Args:
        source: Image path, bytes, BytesIO, or PIL Image.
        convert_rgb: Whether to convert to RGB mode.
        
    Returns:
        PIL Image object.
        
    Raises:
        InvalidImageError: If the image cannot be loaded.
    """
    try:
        if isinstance(source, Image.Image):
            img = source
        elif isinstance(source, bytes):
            img = Image.open(BytesIO(source))
        elif isinstance(source, BytesIO):
            img = Image.open(source)
        elif isinstance(source, (Path, str)):
            path = Path(source)
            if not path.exists():
                raise InvalidImageError(f"File not found: {path}", file_path=str(path))
            img = Image.open(path)
        else:
            raise InvalidImageError(f"Unsupported image source type: {type(source)}")
        
        # Verify it's a valid image
        img.verify()
        
        # Re-open after verify (verify closes the file)
        if isinstance(source, bytes):
            img = Image.open(BytesIO(source))
        elif isinstance(source, BytesIO):
            source.seek(0)
            img = Image.open(source)
        elif isinstance(source, (Path, str)):
            img = Image.open(source)
        
        # Convert to RGB if requested
        if convert_rgb and img.mode != "RGB":
            if img.mode == "RGBA":
                # Handle transparency by compositing on white background
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            else:
                img = img.convert("RGB")
        
        return img
        
    except InvalidImageError:
        raise
    except Exception as e:
        raise InvalidImageError(
            f"Failed to load image: {e}",
            file_path=str(source) if isinstance(source, (Path, str)) else None,
            reason=str(e),
        )


def save_image(
    image: Image.Image,
    output_path: Union[Path, str],
    format: Optional[str] = None,
    quality: int = 95,
) -> Path:
    """
    Save a PIL Image to a file.
    
    Args:
        image: PIL Image to save.
        output_path: Destination path.
        format: Image format (auto-detected if None).
        quality: JPEG quality (1-100).
        
    Returns:
        Path to the saved image.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    save_kwargs = {}
    if format:
        save_kwargs["format"] = format
    
    # Set quality for JPEG
    if output_path.suffix.lower() in [".jpg", ".jpeg"] or format == "JPEG":
        save_kwargs["quality"] = quality
    
    image.save(output_path, **save_kwargs)
    return output_path


def get_image_info(image: Image.Image) -> Dict[str, Any]:
    """
    Extract basic information from a PIL Image.
    
    Args:
        image: PIL Image object.
        
    Returns:
        Dictionary with image information.
    """
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
        "has_alpha": image.mode in ("RGBA", "LA", "PA"),
        "megapixels": (image.width * image.height) / 1_000_000,
    }
