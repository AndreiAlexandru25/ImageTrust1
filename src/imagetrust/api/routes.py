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


@router.post("/analyze/comprehensive")
async def analyze_comprehensive(
    file: UploadFile = File(...),
    include_metadata: bool = Form(default=True),
    include_explainability: bool = Form(default=True),
    include_forensics: bool = Form(default=True),
    include_localization: bool = Form(default=True),
):
    """
    Comprehensive image analysis endpoint for the web frontend.

    Returns full detection results including individual model scores,
    calibrated ensemble, meta-classifier, conformal prediction,
    Grad-CAM heatmaps, forensic analysis, and metadata.
    """
    import time
    import base64
    from io import BytesIO as _BytesIO
    from PIL import Image
    from imagetrust.utils.helpers import timer, load_image

    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}",
        )

    settings = get_settings()
    content = await file.read()
    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB",
        )

    try:
        image = load_image(content)
        analysis_id = generate_id("analysis")
        total_start = time.time()

        # --- Test-Time Adaptive Restoration ---
        restoration_info = None
        try:
            from imagetrust.detection.restoration import adaptive_restore
            restore_result = adaptive_restore(image)
            if restore_result.was_restored:
                image = restore_result.restored_image
                restoration_info = {
                    "was_restored": True,
                    "degradation_type": restore_result.degradation.primary_type.value,
                    "degradation_severity": restore_result.degradation.severity,
                    "steps": restore_result.restoration_applied,
                    "jpeg_quality_est": restore_result.degradation.jpeg_quality_est,
                    "blur_level": restore_result.degradation.blur_level,
                    "noise_level": restore_result.degradation.noise_level,
                }
                logger.info(
                    f"Image restored: {restore_result.restoration_applied}"
                )
        except Exception as e:
            logger.warning(f"Restoration failed: {e}")

        # --- Comprehensive ML Detection (singleton) ---
        from imagetrust.api.main import get_comprehensive_detector

        comprehensive = get_comprehensive_detector()
        comp_result = comprehensive.analyze(
            image, return_uncertainty=True, profile=True,
        )

        # --- Determine confidence level ---
        ai_prob = comp_result["ai_probability"]
        conf = comp_result["confidence"]
        conf_level = Confidence.from_probability(
            max(ai_prob, 1.0 - ai_prob)
        ).value

        # --- Grad-CAM ---
        gradcam_data = None
        gradcam_ms = 0.0
        if include_explainability:
            try:
                gc_start = time.time()
                from imagetrust.explainability.gradcam import (
                    analyze_with_gradcam,
                    heatmap_to_base64,
                )

                gc_result = analyze_with_gradcam(image, use_model=True)
                heatmap_b64 = heatmap_to_base64(
                    Image.fromarray(
                        (gc_result.heatmap * 255).astype("uint8")
                    ).convert("RGB")
                ) if gc_result.heatmap is not None else ""

                overlay_buf = _BytesIO()
                gc_result.overlay.save(overlay_buf, format="PNG")
                overlay_b64 = base64.b64encode(
                    overlay_buf.getvalue()
                ).decode("utf-8")

                gradcam_data = {
                    "heatmap_base64": heatmap_b64,
                    "overlay_base64": overlay_b64,
                    "highlighted_regions": gc_result.highlighted_regions,
                    "activation_score": float(gc_result.activation_score),
                    "layer_name": gc_result.layer_name,
                }
                gradcam_ms = (time.time() - gc_start) * 1000
            except Exception as e:
                logger.warning(f"Grad-CAM failed: {e}")

        # --- Metadata & Provenance ---
        metadata_info = None
        provenance_info = None
        metadata_ms = 0.0
        if include_metadata:
            try:
                meta_start = time.time()
                from imagetrust.metadata.provenance import ProvenanceAnalyzer
                from imagetrust.metadata.exif_parser import EXIFParser

                prov_analyzer = ProvenanceAnalyzer()
                meta_analysis, prov_analysis = prov_analyzer.analyze(
                    content, filename=file.filename,
                )

                metadata_info = {
                    "has_exif": meta_analysis.has_metadata
                    and meta_analysis.exif is not None,
                    "ai_indicators": meta_analysis.ai_indicators,
                    "anomalies": meta_analysis.anomalies,
                    "file_name": meta_analysis.file_name,
                    "file_size": meta_analysis.file_size,
                    "width": meta_analysis.width,
                    "height": meta_analysis.height,
                    "format": meta_analysis.format,
                }

                if meta_analysis.exif:
                    exif = meta_analysis.exif
                    metadata_info["exif"] = {
                        "make": exif.make,
                        "model": exif.model,
                        "software": exif.software,
                        "datetime_original": (
                            str(exif.datetime_original)
                            if exif.datetime_original
                            else None
                        ),
                        "exposure_time": exif.exposure_time,
                        "f_number": exif.f_number,
                        "iso": exif.iso,
                        "focal_length": exif.focal_length,
                        "has_camera_info": exif.has_camera_info,
                    }

                provenance_info = {
                    "status": prov_analysis.status.value,
                    "trust_indicators": prov_analysis.trust_indicators,
                    "warning_indicators": prov_analysis.warning_indicators,
                    "claimed_source": prov_analysis.claimed_source,
                    "claimed_creator": prov_analysis.claimed_creator,
                    "creation_date": (
                        str(prov_analysis.creation_date)
                        if prov_analysis.creation_date
                        else None
                    ),
                    "confidence_score": prov_analysis.confidence_score,
                }
                metadata_ms = (time.time() - meta_start) * 1000
            except Exception as e:
                logger.warning(f"Metadata extraction failed: {e}")

        # --- Forensics ---
        forensics_data = None
        if include_forensics:
            try:
                evidence = []

                # Check for copy-move
                copy_move = False
                try:
                    from imagetrust.detection.copy_move_detector import (
                        CopyMoveDetector,
                    )
                    cm = CopyMoveDetector()
                    cm_result = cm.detect(image)
                    copy_move = cm_result.get("detected", False)
                    if copy_move:
                        evidence.append({
                            "type": "Copy-Move",
                            "severity": "critical",
                            "description": "Copy-move forgery detected",
                            "details": cm_result.get("details", ""),
                        })
                except Exception:
                    pass

                # Build forensic evidence from signals
                for res in comp_result.get("individual_results", []):
                    if res["ai_probability"] > 0.7:
                        severity = (
                            "critical" if res["ai_probability"] > 0.85
                            else "warning"
                        )
                        evidence.append({
                            "type": res["method"],
                            "severity": severity,
                            "description": (
                                f"High AI probability ({res['ai_probability']:.1%})"
                                f" detected by {res['method']}"
                            ),
                        })
                    elif res["ai_probability"] < 0.3:
                        evidence.append({
                            "type": res["method"],
                            "severity": "info",
                            "description": (
                                f"Low AI probability ({res['ai_probability']:.1%})"
                                f" - appears authentic per {res['method']}"
                            ),
                        })

                auth_score = 1.0 - ai_prob
                # NOTE: primary_verdict is set to a placeholder here;
                # it will be overwritten after the final verdict override
                # logic runs (see below).
                forensics_data = {
                    "primary_verdict": "pending",
                    "authenticity_score": auth_score,
                    "evidence": evidence,
                    "copy_move_detected": copy_move,
                    "splicing_detected": False,
                }
            except Exception as e:
                logger.warning(f"Forensics failed: {e}")

        # --- Patch-Level AI Localization ---
        localization_data = None
        localization_ms = 0.0
        if include_localization:
            try:
                loc_start = time.time()
                from imagetrust.detection.patch_localizer import (
                    localize_ai_regions,
                )

                loc_result = localize_ai_regions(
                    image, patch_size=128, stride=64,
                )
                if not loc_result.get("skipped"):
                    localization_data = {
                        "heatmap_base64": loc_result["heatmap_base64"],
                        "overlay_base64": loc_result["overlay_base64"],
                        "grid_shape": loc_result["grid_shape"],
                        "hot_regions": loc_result["hot_regions"],
                        "mean_ai_prob": loc_result["mean_ai_prob"],
                        "max_ai_prob": loc_result["max_ai_prob"],
                        "n_patches": loc_result["n_patches"],
                        "n_models_used": loc_result["n_models_used"],
                    }
                localization_ms = (time.time() - loc_start) * 1000
            except Exception as e:
                logger.warning(f"Patch localization failed: {e}")

        # --- Screenshot Detection ---
        screenshot_data = None
        screenshot_ms = 0.0
        try:
            ss_start = time.time()
            from imagetrust.detection.screenshot_detector import detect_screenshot

            exif_dict = None
            file_fmt = None
            if metadata_info:
                exif_dict = metadata_info.get("exif")
                file_fmt = metadata_info.get("format")

            ss_result = detect_screenshot(
                image, exif_data=exif_dict, file_format=file_fmt,
                filename=file.filename,
            )
            screenshot_data = {
                "is_screenshot": ss_result.is_screenshot,
                "probability": ss_result.probability,
                "confidence": ss_result.confidence,
                "indicators": ss_result.indicators,
            }
            screenshot_ms = (time.time() - ss_start) * 1000
        except Exception as e:
            logger.warning(f"Screenshot detection failed: {e}")

        # --- Determine final verdict (with full override logic) ---
        raw_ai_prob = ai_prob
        final_verdict = comp_result.get("verdict", "uncertain")
        raw_verdict = final_verdict
        indiv = comp_result.get("individual_results", [])
        override_reasons = []

        # Extract HF model probabilities (external pre-trained detectors)
        hf_entries = [
            r for r in indiv
            if r.get("method", "").startswith("ML:")
            and "calibrated" not in r.get("method", "").lower()
            and "Ensemble" not in r.get("method", "")
            and "Custom Trained" not in r.get("method", "")
        ]
        hf_probs = [r["ai_probability"] for r in hf_entries]
        hf_ai_avg = sum(hf_probs) / len(hf_probs) if hf_probs else 0.5
        hf_ai_votes = sum(1 for p in hf_probs if p > 0.5)

        # Extract CNN calibrated model probabilities
        cnn_entries = [
            r for r in indiv
            if "calibrated" in r.get("method", "").lower()
            and "Ensemble" not in r.get("method", "")
        ]
        cnn_probs = [r["ai_probability"] for r in cnn_entries]
        cnn_avg = sum(cnn_probs) / len(cnn_probs) if cnn_probs else 0.5

        # Extract signal analysis probabilities
        signal_entries = [
            r for r in indiv
            if not r.get("method", "").startswith("ML:")
        ]
        signal_probs = [r["ai_probability"] for r in signal_entries]
        signal_avg = (
            sum(signal_probs) / len(signal_probs)
            if signal_probs else 0.5
        )

        # Override 1: Strong HF model consensus overrides when
        # >=3 out of 4 HF models agree on AI with high avg probability.
        # This catches cases where CNN models miss AI-generated content
        # that external pre-trained detectors catch.
        if final_verdict == "real" and len(hf_probs) >= 3:
            hf_vote_ratio = hf_ai_votes / len(hf_probs)
            if hf_ai_avg > 0.75 and hf_vote_ratio >= 0.75:
                final_verdict = "ai_generated"
                ai_prob = max(ai_prob, hf_ai_avg * 0.85)
                override_reasons.append(
                    f"Strong HF model consensus: {hf_ai_votes}/{len(hf_probs)} "
                    f"pre-trained detectors voted AI (avg {hf_ai_avg * 100:.1f}%)"
                )

        # Override 2: Screenshot detection (based on image content analysis)
        if screenshot_data and screenshot_data["is_screenshot"]:
            ss_prob = screenshot_data["probability"]
            if ss_prob > 0.65 and ai_prob < 0.75:
                final_verdict = "screenshot"
                override_reasons.append(
                    f"Screenshot/screen capture detected via image "
                    f"characteristics (probability {ss_prob * 100:.0f}%)"
                )

        # Override 3: Patch-level localization found AI-manipulated regions.
        # This is the most reliable evidence for partial AI edits (e.g. a
        # real photograph with AI-added elements) because CNN/HF models are
        # trained on fully-generated images and miss localised manipulation.
        # A single critical region with z-score > 4.0 is statistically
        # significant (p < 10^-5) – the patch is genuinely different from
        # the rest of the image.
        if final_verdict == "real" and localization_data:
            hot = localization_data.get("hot_regions", [])
            critical = [
                r for r in hot if r.get("severity") == "critical"
            ]
            warning_regions = [
                r for r in hot if r.get("severity") == "warning"
            ]

            if critical:
                max_zscore = max(r["z_score"] for r in critical)
                max_ai_prob_loc = max(
                    r["ai_probability"] for r in critical
                )
                n_hot = len(critical) + len(warning_regions)

                if max_zscore > 4.0:
                    # Strong statistical outlier – localized AI content
                    final_verdict = "manipulated"
                    # Probability blends localization evidence:
                    # 60% from max patch AI prob + 30% from z-score strength
                    loc_evidence = (
                        max_ai_prob_loc * 0.6
                        + min(max_zscore, 10.0) / 10.0 * 0.3
                    )
                    ai_prob = max(ai_prob, loc_evidence)
                    override_reasons.append(
                        f"Patch-level localization detected {len(critical)} "
                        f"critical AI region(s) with z-score {max_zscore:.1f} "
                        f"(max AI prob {max_ai_prob_loc:.1%}), indicating "
                        f"localised AI manipulation"
                    )
                elif len(critical) >= 2 or n_hot >= 3:
                    # Multiple suspicious regions but lower statistical
                    # significance – flag as uncertain
                    final_verdict = "uncertain"
                    ai_prob = max(ai_prob, max_ai_prob_loc * 0.5)
                    override_reasons.append(
                        f"Patch-level localization found {n_hot} suspicious "
                        f"region(s) – possible manipulation"
                    )

        # --- Build override metadata ---
        override_applied = final_verdict != raw_verdict
        override_reason = None
        if override_applied:
            override_reason = "; ".join(override_reasons) if override_reasons else None

        # --- Recompute votes to exclude signal analysis ---
        # Signal analyzers (frequency, texture, noise, edge, color) have
        # tiny weights (0.03-0.05) and almost always vote "Real", diluting
        # the meaningful ML model votes. Only count ML models in the bar.
        ml_only = [
            r for r in indiv
            if r.get("method", "").startswith("ML:")
        ]
        ml_votes_ai = sum(
            1 for r in ml_only if r.get("ai_probability", 0) > 0.5
        )
        ml_votes_real = len(ml_only) - ml_votes_ai
        votes = {
            "ai": ml_votes_ai,
            "real": ml_votes_real,
            "total": len(ml_only),
        }

        # --- Adjust confidence when override contradicts model votes ---
        if override_applied and votes.get("total", 0) > 0:
            vote_agreement = votes.get("ai", 0) / votes["total"]
            if final_verdict in ("ai_generated", "manipulated"):
                if vote_agreement < 0.5:
                    conf = conf * (0.5 + vote_agreement)

        # --- Build detection_summary for frontend ---
        ai_signal_count = sum(1 for p in signal_probs if p > 0.5)
        ai_hf_count = sum(1 for p in hf_probs if p > 0.5)
        ai_cnn_count = sum(1 for p in cnn_probs if p > 0.5)

        # Localization summary
        loc_max_ai = 0.0
        loc_max_zscore = 0.0
        loc_critical_count = 0
        loc_warning_count = 0
        if localization_data:
            hot = localization_data.get("hot_regions", [])
            if hot:
                loc_max_ai = max(r["ai_probability"] for r in hot)
                loc_max_zscore = max(r.get("z_score", 0) for r in hot)
                loc_critical_count = sum(
                    1 for r in hot if r.get("severity") == "critical"
                )
                loc_warning_count = sum(
                    1 for r in hot if r.get("severity") == "warning"
                )

        detection_summary = {
            "raw_verdict": raw_verdict,
            "raw_ai_probability": raw_ai_prob,
            "cnn_avg": cnn_avg,
            "cnn_verdict": "ai_generated" if cnn_avg > 0.5 else "real",
            "hf_avg": hf_ai_avg,
            "hf_verdict": "ai_generated" if hf_ai_avg > 0.5 else "real",
            "hf_ai_count": ai_hf_count,
            "hf_total": len(hf_probs),
            "signal_avg": signal_avg,
            "signal_ai_count": ai_signal_count,
            "signal_total": len(signal_probs),
            "cnn_ai_count": ai_cnn_count,
            "cnn_total": len(cnn_probs),
            "loc_max_ai_prob": loc_max_ai,
            "loc_max_zscore": loc_max_zscore,
            "loc_critical_count": loc_critical_count,
            "loc_warning_count": loc_warning_count,
            "models_agree_with_verdict": (
                (final_verdict in ("ai_generated", "manipulated")
                 and (votes.get("ai", 0) > votes.get("real", 0)
                      or loc_critical_count > 0))
                or (final_verdict == "real"
                    and votes.get("real", 0) > votes.get("ai", 0))
                or final_verdict in ("uncertain", "screenshot")
            ),
        }

        total_ms = (time.time() - total_start) * 1000

        # --- Update forensics with final overridden verdict ---
        if forensics_data:
            forensics_data["primary_verdict"] = final_verdict
            forensics_data["authenticity_score"] = 1.0 - ai_prob
            # Add override evidence to forensics
            if override_applied and override_reasons:
                for reason in override_reasons:
                    forensics_data["evidence"].insert(0, {
                        "type": "Verdict Override",
                        "severity": "critical",
                        "description": reason,
                        "details": (
                            f"Raw verdict '{raw_verdict}' was overridden "
                            f"to '{final_verdict}' based on additional "
                            f"evidence beyond ML model predictions."
                        ),
                    })

        # --- Timing breakdown ---
        timing = comp_result.get("timing_breakdown", {})
        if gradcam_ms > 0:
            timing["gradcam_ms"] = gradcam_ms
        if metadata_ms > 0:
            timing["metadata_ms"] = metadata_ms
        if localization_ms > 0:
            timing["localization_ms"] = localization_ms
        if screenshot_ms > 0:
            timing["screenshot_ms"] = screenshot_ms
        timing["total_ms"] = total_ms

        # --- Build response ---
        response = {
            "analysis_id": analysis_id,
            "verdict": final_verdict,
            "ai_probability": ai_prob,
            "confidence": conf,
            "confidence_level": conf_level,
            "override_applied": override_applied,
            "override_reason": override_reason,
            "raw_verdict": raw_verdict,
            "raw_ai_probability": raw_ai_prob,
            "detection_summary": detection_summary,
            "votes": votes,
            "individual_results": comp_result.get("individual_results", []),
            "calibrated_ensemble": comp_result.get("calibrated_ensemble"),
            "uncertainty": comp_result.get("uncertainty"),
            "meta_classifier": comp_result.get("meta_classifier"),
            "conformal_prediction": comp_result.get("conformal_prediction"),
            "gradcam": gradcam_data,
            "forensics": forensics_data,
            "metadata": metadata_info,
            "provenance": provenance_info,
            "localization": localization_data,
            "screenshot": screenshot_data,
            "restoration": restoration_info,
            "timing_breakdown": timing,
            "processing_time_ms": total_ms,
            "image_dimensions": [image.width, image.height],
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
