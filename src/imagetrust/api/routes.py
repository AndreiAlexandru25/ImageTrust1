"""
API routes for ImageTrust.
"""

from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from imagetrust.core.config import get_settings
from imagetrust.core.types import DetectionVerdict, Confidence
from imagetrust.utils.helpers import generate_id
from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["detection"])


# Pydantic models for API
class DetectionResult(BaseModel):
    """Detection result response."""
    analysis_id: str
    timestamp: datetime
    ai_probability: float
    real_probability: float
    verdict: str
    confidence: str
    calibrated: bool
    processing_time_ms: float
    model_name: str


class BatchDetectionResult(BaseModel):
    """Batch detection result response."""
    results: List[DetectionResult]
    total_images: int
    total_time_ms: float


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    device: str
    input_size: int
    is_ensemble: bool
    calibrated: bool


@router.post("/analyze", response_model=DetectionResult)
async def analyze_image(
    file: UploadFile = File(...),
    include_metadata: bool = Form(default=False),
    include_explainability: bool = Form(default=False),
):
    """
    Analyze an image for AI-generated content.
    
    - **file**: Image file to analyze
    - **include_metadata**: Include metadata analysis
    - **include_explainability**: Include Grad-CAM and patch analysis
    """
    from imagetrust.api.main import get_detector
    from imagetrust.utils.helpers import timer, load_image
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )
    
    # Check file size
    settings = get_settings()
    content = await file.read()
    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB"
        )
    
    try:
        # Load image
        image = load_image(content)
        
        # Get detector
        detector = get_detector()
        
        # Analyze
        with timer() as t:
            result = detector.detect(image)
        
        # Build response
        response = DetectionResult(
            analysis_id=generate_id("analysis"),
            timestamp=datetime.utcnow(),
            ai_probability=result["ai_probability"],
            real_probability=result["real_probability"],
            verdict=result["verdict"].value,
            confidence=result["confidence"].value,
            calibrated=result["calibrated"],
            processing_time_ms=t["elapsed_ms"],
            model_name=result["model_name"],
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/batch", response_model=BatchDetectionResult)
async def analyze_batch(
    files: List[UploadFile] = File(...),
):
    """
    Analyze multiple images in batch.
    
    - **files**: List of image files to analyze
    """
    from imagetrust.api.main import get_detector
    from imagetrust.utils.helpers import timer, load_image
    
    if len(files) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 images per batch"
        )
    
    detector = get_detector()
    results = []
    
    with timer() as total_time:
        for file in files:
            try:
                content = await file.read()
                image = load_image(content)
                
                with timer() as t:
                    result = detector.detect(image)
                
                results.append(DetectionResult(
                    analysis_id=generate_id("analysis"),
                    timestamp=datetime.utcnow(),
                    ai_probability=result["ai_probability"],
                    real_probability=result["real_probability"],
                    verdict=result["verdict"].value,
                    confidence=result["confidence"].value,
                    calibrated=result["calibrated"],
                    processing_time_ms=t["elapsed_ms"],
                    model_name=result["model_name"],
                ))
                
            except Exception as e:
                logger.warning(f"Failed to process {file.filename}: {e}")
    
    return BatchDetectionResult(
        results=results,
        total_images=len(files),
        total_time_ms=total_time["elapsed_ms"],
    )


@router.get("/model", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    from imagetrust.api.main import get_detector
    
    detector = get_detector()
    info = detector.get_model_info()
    
    return ModelInfo(
        model_name=info.get("model_name", "unknown"),
        device=info.get("device", "unknown"),
        input_size=info.get("input_size", 224),
        is_ensemble=info.get("is_ensemble", False),
        calibrated=info.get("calibrated", False),
    )
