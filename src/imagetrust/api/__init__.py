"""
ImageTrust API Module
=====================
FastAPI-based REST API for AI detection.
"""

from imagetrust.api.main import app, create_app
from imagetrust.api.routes import router

__all__ = [
    "app",
    "create_app",
    "router",
]
