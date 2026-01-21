"""
Logging configuration for ImageTrust.

Uses `loguru` for flexible and user-friendly logging.
"""

import sys
import os
from pathlib import Path
from typing import Literal, Optional

from loguru import logger

# Remove default handler
logger.remove()

# Track if logging has been set up
_logging_configured = False


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    colorize: bool = True,
) -> None:
    """
    Set up global logging configuration.

    Args:
        level: Minimum logging level to display.
        log_file: Optional path to a file for logging.
        rotation: Log file rotation policy (e.g., "10 MB", "1 day").
        retention: Log file retention policy (e.g., "7 days", "1 month").
        colorize: Whether to colorize console output.
    """
    global _logging_configured
    
    # Remove existing handlers if reconfiguring
    logger.remove()

    # Console format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    # File format (no colors)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}"
    )

    # Add console handler if available (PyInstaller --noconsole may set stderr to None)
    stderr_available = sys.stderr is not None and hasattr(sys.stderr, "write")
    if stderr_available:
        logger.add(
            sink=sys.stderr,
            level=level,
            colorize=colorize,
            format=console_format,
        )
    elif log_file is None:
        # Fallback to a local file log when no console is available
        appdata = os.getenv("APPDATA")
        base_dir = Path(appdata) if appdata else Path.home()
        log_file = base_dir / "ImageTrust" / "logs" / "imagetrust.log"

    # Add file handler if specified
    if log_file:
        try:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                sink=str(log_file),
                level=level,
                rotation=rotation,
                retention=retention,
                compression="zip",
                format=file_format,
            )
            logger.info(f"File logging enabled: {log_file}")
        except Exception:
            # If file logging fails, skip without crashing the app
            pass

    _logging_configured = True
    logger.debug(f"Logging configured with level: {level}")


def get_logger(name: str = None):
    """
    Get a logger instance.

    Args:
        name: Optional name for the logger (typically __name__).

    Returns:
        A loguru logger instance, optionally bound with the module name.
    """
    global _logging_configured
    
    # Auto-configure if not done
    if not _logging_configured:
        setup_logging()
    
    if name:
        return logger.bind(name=name)
    return logger


# Convenience exports
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical
exception = logger.exception
