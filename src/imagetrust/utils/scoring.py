"""
Shared scoring utilities for ImageTrust.
Keeps the scoring logic consistent across frontend and desktop app.
"""

from typing import Optional
import numpy as np
from PIL import Image


def analyze_image_source(image: Image.Image, image_bytes: bytes, uploaded_file, metadata: dict) -> dict:
    """Infer source/compression characteristics (WhatsApp/Instagram/etc.)."""
    width, height = image.size
    filename_lower = uploaded_file.name.lower() if uploaded_file else ""
    ext = filename_lower.split(".")[-1] if "." in filename_lower else ""
    is_jpeg = ext in ["jpg", "jpeg"]

    # Bytes-per-pixel (low values indicate heavy compression)
    if image_bytes and width > 0 and height > 0:
        bpp = len(image_bytes) / (width * height)
    else:
        bpp = 0.0

    if bpp < 0.25:
        compression_level = "high"
    elif bpp < 0.45:
        compression_level = "medium"
    else:
        compression_level = "low"

    social_keywords = ["whatsapp", "instagram", "insta", "facebook", "messenger", "telegram", "snap"]
    platform = next((k for k in social_keywords if k in filename_lower), None)
    is_social = platform is not None

    # Heuristic: JPEG + low bpp + missing EXIF => likely compressed share
    missing_exif = not metadata.get("has_exif")
    likely_compressed = is_jpeg and (compression_level != "low") and missing_exif

    return {
        "filename": filename_lower,
        "extension": ext,
        "bpp": bpp,
        "compression_level": compression_level,
        "is_social": is_social,
        "platform": platform,
        "likely_compressed": likely_compressed,
        "is_jpeg": is_jpeg,
        "missing_exif": missing_exif,
    }


def compute_combined_score(
    result: dict,
    uploaded_file,
    source_info: Optional[dict] = None,
    settings: Optional[dict] = None,
) -> dict:
    """Compute the unified AI probability score used across the UI."""
    settings = settings or {}
    use_ml = settings.get("use_ml", True)
    use_frequency = settings.get("use_frequency", True)
    use_noise = settings.get("use_noise", True)
    auto_calibration = settings.get("auto_calibration", True)

    # Filter active results based on settings
    active_results = []
    for r in result.get("individual_results", []):
        method = r.get("method", "")
        if method.startswith("ML:") and use_ml:
            active_results.append(r)
        elif "Frequency Analysis" in method and use_frequency:
            active_results.append(r)
        elif "Noise Pattern" in method and use_noise:
            active_results.append(r)

    ml_entries = [r for r in active_results if r["method"].startswith("ML:")]
    signal_entries = [r for r in active_results if not r["method"].startswith("ML:")]

    # Default source info
    source_info = source_info or {
        "is_social": False,
        "likely_compressed": False,
        "compression_level": "low",
        "platform": None,
    }

    # Apply social-media calibration
    calibration_applied = False
    calibration_note = ""
    signal_suppressed = False
    if auto_calibration and (source_info.get("is_social") or source_info.get("likely_compressed")):
        calibration_applied = True
        platform = source_info.get("platform") or "social media"
        compression = source_info.get("compression_level", "unknown")
        calibration_note = f"Auto calibration applied for {platform} (compression: {compression})."

        # Suppress signal analysis on compressed images
        if source_info.get("compression_level") in ["high", "medium"]:
            signal_entries = []
            active_results = [r for r in active_results if r["method"].startswith("ML:")]
            signal_suppressed = True

    # ML weighted average (robust to outliers on social media)
    ml_probs = [r["ai_probability"] for r in ml_entries]
    ml_methods = [r["method"] for r in ml_entries]

    def weighted_ml_average(probs: list, methods: list, profile: str) -> float:
        if not probs:
            return 0.5
        if profile == "social":
            weights_map = {
                "ML: Deepfake vs Real": 0.35,
                "ML: AI Image Detector": 0.15,
                "ML: AIorNot Detector": 0.10,
                "ML: NYUAD Detector (2025)": 0.40,
            }
        else:
            weights_map = {
                "ML: Deepfake vs Real": 0.25,
                "ML: AI Image Detector": 0.25,
                "ML: AIorNot Detector": 0.25,
                "ML: NYUAD Detector (2025)": 0.25,
            }

        weights = []
        for m in methods:
            weights.append(weights_map.get(m, 0.25))

        total_w = sum(weights) or 1.0
        return float(sum(p * w for p, w in zip(probs, weights)) / total_w)

    profile = "social" if calibration_applied else "default"
    ml_avg = weighted_ml_average(ml_probs, ml_methods, profile)

    signal_results = [r["ai_probability"] for r in signal_entries]
    signal_avg = np.mean(signal_results) if signal_results else 0.5

    # Voting based on active results
    all_results = [r["ai_probability"] for r in active_results]
    votes_ai = sum(1 for p in all_results if p > 0.5)
    vote_ratio = votes_ai / len(all_results) if all_results else 0.5

    # Advanced AI detection (avoid for social media)
    file_ext = uploaded_file.name.lower().split('.')[-1] if uploaded_file else ""
    is_png = file_ext == "png"
    filename_lower = uploaded_file.name.lower() if uploaded_file else ""
    chatgpt_filename = "chatgpt" in filename_lower or "dall" in filename_lower

    ml_max = max(ml_probs) if ml_probs else 0.5
    ml_consensus_real = ml_max < 0.40
    advanced_ai_detected = is_png and ml_consensus_real and signal_avg > 0.60
    advanced_mode = (advanced_ai_detected or (chatgpt_filename and signal_avg > 0.50)) and not calibration_applied

    # Combine scores
    if not signal_results:
        combined = (ml_avg * 0.90 + vote_ratio * 0.10)
        weights = {"ml": "90%", "signal": "0%", "vote": "10%"}
    elif advanced_mode:
        combined = (ml_avg * 0.20 + signal_avg * 0.60 + vote_ratio * 0.20)
        if chatgpt_filename:
            combined = min(combined + 0.15, 0.95)
        elif signal_avg > 0.65:
            combined = min(combined + 0.10, 0.90)
        weights = {"ml": "20%", "signal": "60%", "vote": "20%"}
    else:
        combined = (ml_avg * 0.70 + signal_avg * 0.20 + vote_ratio * 0.10)
        weights = {"ml": "70%", "signal": "20%", "vote": "10%"}

    # Social-media stabilization: reduce false positives
    if calibration_applied and ml_probs:
        ml_spread = max(ml_probs) - min(ml_probs)
        if ml_spread > 0.6 and ml_avg < 0.60:
            combined = min(combined, 0.35)
        ml_ai_votes = sum(1 for p in ml_probs if p > 0.70)
        strong_ai = ml_ai_votes >= 3 and ml_avg >= 0.70
        if not strong_ai:
            combined = min(combined, 0.20)

    return {
        "combined": combined,
        "ai_prob": combined,
        "real_prob": 1 - combined,
        "ml_avg": ml_avg,
        "signal_avg": signal_avg,
        "ml_results": ml_probs,
        "signal_results": signal_results,
        "votes_ai": votes_ai,
        "vote_ratio": vote_ratio,
        "all_results": all_results,
        "advanced_mode": advanced_mode,
        "advanced_ai_detected": advanced_ai_detected,
        "chatgpt_filename": chatgpt_filename,
        "is_png": is_png,
        "weights": weights,
        "active_results": active_results,
        "calibration_applied": calibration_applied,
        "calibration_note": calibration_note,
        "signal_suppressed": signal_suppressed,
        "source_info": source_info,
    }
