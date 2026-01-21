"""
FastAPI application for ImageTrust.
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from imagetrust.core.config import get_settings
from imagetrust.api.routes import router
from imagetrust.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Global detector instance
_detector = None


def get_detector():
    """Get the global detector instance."""
    global _detector
    if _detector is None:
        from imagetrust.detection import AIDetector
        settings = get_settings()
        _detector = AIDetector(
            model="ensemble" if settings.ensemble_enabled else settings.detector_backbone,
        )
    return _detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting ImageTrust API...")
    settings = get_settings()
    setup_logging(level=settings.log_level)
    
    # Pre-load detector
    get_detector()
    logger.info("Detector loaded successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ImageTrust API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="ImageTrust API",
        description="AI-Generated Image Detection API",
        version=settings.project_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router)
    
    return app


# Create app instance
app = create_app()


@app.get("/")
async def root():
    """Root endpoint."""
    settings = get_settings()
    return {
        "name": settings.project_name,
        "version": settings.project_version,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
