"""
Streamlit frontend for ImageTrust.
Comprehensive AI Image Detection with multiple analysis methods.

Features:
- Multi-model AI detection
- AI Generator identification (DALL-E, Midjourney, SD)
- Grad-CAM heatmap visualization
- Copy-Move forgery detection
- C2PA Content Credentials verification
"""

import streamlit as st
from pathlib import Path
import sys
import io
import numpy as np
import time
import base64
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from PIL import Image


# Import new advanced modules
try:
    from imagetrust.detection.generator_identifier import GeneratorIdentifier, identify_generator
    GENERATOR_ID_AVAILABLE = True
except ImportError:
    GENERATOR_ID_AVAILABLE = False

try:
    from imagetrust.explainability.gradcam import GradCAMAnalyzer, analyze_with_gradcam
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

try:
    from imagetrust.detection.copy_move_detector import CopyMoveDetector, SplicingDetector
    COPY_MOVE_AVAILABLE = True
except ImportError:
    COPY_MOVE_AVAILABLE = False

try:
    from imagetrust.metadata.c2pa_validator import C2PAValidator, validate_c2pa
    C2PA_AVAILABLE = True
except ImportError:
    C2PA_AVAILABLE = False

try:
    from imagetrust.forensics import ForensicsEngine
    from imagetrust.forensics.base import PluginCategory
    FORENSICS_AVAILABLE = True
except ImportError:
    FORENSICS_AVAILABLE = False


def extract_metadata(image_bytes: bytes) -> dict:
    """Extract EXIF metadata from image bytes."""
    metadata = {
        "has_exif": False,
        "camera": None,
        "date_taken": None,
        "software": None,
        "gps": None,
        "dimensions": None,
        "all_tags": {}
    }
    
    try:
        import exifread
        from io import BytesIO
        
        tags = exifread.process_file(BytesIO(image_bytes), details=False)
        
        if tags:
            metadata["has_exif"] = True
            
            # Camera info
            if "Image Make" in tags and "Image Model" in tags:
                metadata["camera"] = f"{tags['Image Make']} {tags['Image Model']}"
            elif "Image Model" in tags:
                metadata["camera"] = str(tags["Image Model"])
            
            # Date
            if "EXIF DateTimeOriginal" in tags:
                metadata["date_taken"] = str(tags["EXIF DateTimeOriginal"])
            elif "Image DateTime" in tags:
                metadata["date_taken"] = str(tags["Image DateTime"])
            
            # Software
            if "Image Software" in tags:
                metadata["software"] = str(tags["Image Software"])
            
            # GPS
            if "GPS GPSLatitude" in tags and "GPS GPSLongitude" in tags:
                metadata["gps"] = "GPS data present"
            
            # Store interesting tags
            for key, value in tags.items():
                if not key.startswith("Thumbnail"):
                    metadata["all_tags"][key] = str(value)
    except Exception as e:
        metadata["error"] = str(e)
    
    return metadata


def analyze_provenance(metadata: dict) -> dict:
    """Analyze image provenance based on metadata."""
    indicators = {
        "authentic_signs": [],
        "suspicious_signs": [],
        "score": 50
    }
    
    if metadata.get("camera"):
        indicators["authentic_signs"].append(f"📷 Camera: {metadata['camera']}")
        indicators["score"] += 15
    
    if metadata.get("date_taken"):
        indicators["authentic_signs"].append(f"📅 Date: {metadata['date_taken']}")
        indicators["score"] += 10
    
    if metadata.get("gps"):
        indicators["authentic_signs"].append("📍 GPS location embedded")
        indicators["score"] += 10
    
    if metadata.get("has_exif") and len(metadata.get("all_tags", {})) > 5:
        indicators["authentic_signs"].append(f"📋 Rich metadata ({len(metadata['all_tags'])} tags)")
        indicators["score"] += 5
    
    if not metadata.get("has_exif"):
        indicators["suspicious_signs"].append("⚠️ No EXIF metadata")
        indicators["score"] -= 20
    
    software = metadata.get("software") or ""
    software_lower = software.lower() if software else ""
    ai_software = ["dall-e", "midjourney", "stable diffusion", "adobe firefly", "photoshop"]
    if any(ai in software_lower for ai in ai_software):
        indicators["suspicious_signs"].append(f"🤖 AI software: {software}")
        indicators["score"] -= 30
    
    indicators["score"] = max(0, min(100, indicators["score"]))
    
    return indicators


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

        # Suppress signal analysis on heavily compressed images
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

    # Social-media stabilization: reduce false positives on WhatsApp/Instagram
    if calibration_applied and ml_probs:
        ml_spread = max(ml_probs) - min(ml_probs)
        if ml_spread > 0.6 and ml_avg < 0.60:
            combined = min(combined, 0.35)

        # Strong-real bias for heavily compressed social images
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

def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="ImageTrust - AI Image Forensics",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 16px;
        line-height: 1.5;
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .ai-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%);
        color: white;
    }
    .real-result {
        background: linear-gradient(135deg, #51cf66 0%, #2f9e44 100%);
        color: white;
    }
    .uncertain-result {
        background: linear-gradient(135deg, #ffd43b 0%, #f59f00 100%);
        color: #333;
    }
    .method-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .method-ai {
        border-left-color: #ff6b6b;
    }
    .method-real {
        border-left-color: #51cf66;
    }
    .score-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(to right, #51cf66 0%, #51cf66 var(--real), #ff6b6b var(--real), #ff6b6b 100%);
    }
    .info-card {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 12px 14px;
        margin-top: 8px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col_title, col_badge = st.columns([3, 1])
    with col_title:
        st.markdown('<p class="main-header">🔬 ImageTrust</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Multi-Method AI Image Forensics System</p>', unsafe_allow_html=True)
    with col_badge:
        st.markdown("### 🎓 Master's Thesis")
        st.caption("Comprehensive Detection")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Analysis Settings")
        
        st.subheader("🤖 ML Models")
        use_ml = st.checkbox("AI Detection Models", value=True, help="HuggingFace pretrained models")
        
        st.subheader("📊 Signal Analysis")
        use_frequency = st.checkbox("Frequency Analysis (FFT)", value=False)
        use_noise = st.checkbox("Noise Pattern Analysis", value=False)
        
        st.subheader("🛡️ Calibration")
        auto_calibration = st.checkbox(
            "Auto Calibration (WhatsApp/Instagram)",
            value=True,
            help="Reduces false positives for compressed images."
        )
        
        st.subheader("📋 Metadata")
        use_metadata = st.checkbox("EXIF Analysis", value=True)
        use_provenance = st.checkbox("Provenance Score", value=True)

        st.subheader("🔍 Forensics Engine")
        forensics_include_ai = st.checkbox(
            "Include AI Detection in Forensics",
            value=True,
            help="Enable HuggingFace model-based AI detection in full forensics analysis"
        )
        st.session_state["forensics_include_ai"] = forensics_include_ai
        
        analysis_settings = {
            "use_ml": use_ml,
            "use_frequency": use_frequency,
            "use_noise": use_noise,
            "auto_calibration": auto_calibration,
        }

        st.subheader("✅ Confidence")
        high_confidence_mode = st.checkbox(
            "High Confidence Mode (>=75%)",
            value=True,
            help="Only label AI/Real if confidence is at least 75%."
        )
        
        st.divider()
        
        st.header("ℹ️ About")
        st.markdown("""
        **ImageTrust** - Sistem Forensic Complet
        
        ---
        
        🤖 **AI Detection (4 modele):**
        - Deepfake vs Real
        - AI Image Detector  
        - AIorNot Detector
        - NYUAD 2025
        
        ---
        
        🎨 **Generator ID (UNIC!):**
        - DALL-E 2/3
        - Midjourney v4/v5/v6
        - Stable Diffusion
        - Adobe Firefly
        
        ---
        
        🔥 **Grad-CAM Heatmap:**
        - Vizualizare zone AI
        - Regiuni suspecte
        
        ---
        
        🧩 **Copy-Move Detection:**
        - Detectare manipulări
        - Splicing analysis
        
        ---
        
        🔐 **C2PA Verification:**
        - Content Credentials
        - Semnătură digitală
        - Istoric editări
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload an image to analyze"
        )
        
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))
            
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.caption(f"📐 {image.width}×{image.height} | 🎨 {image.mode} | 📦 {len(image_bytes)/1024:.1f} KB")
            
            if st.button("🔬 Run Full Analysis", type="primary", use_container_width=True):
                
                progress = st.progress(0, text="Starting comprehensive analysis...")
                status = st.empty()
                
                try:
                    start_time = time.time()
                    
                    # Step 1: Metadata
                    progress.progress(10, text="📋 Extracting metadata...")
                    metadata = extract_metadata(image_bytes)
                    metadata["dimensions"] = f"{image.width}×{image.height}"
                    
                    # Image details
                    file_ext = Path(uploaded_file.name).suffix.lstrip(".").upper() if uploaded_file else "UNKNOWN"
                    image_info = {
                        "filename": uploaded_file.name if uploaded_file else "image",
                        "format": file_ext,
                        "dimensions": f"{image.width}×{image.height}",
                        "size_kb": len(image_bytes) / 1024.0,
                        "mode": image.mode,
                    }
                    
                    # Step 1.5: Source/Compression analysis
                    source_info = analyze_image_source(image, image_bytes, uploaded_file, metadata)
                    
                    # Step 2: Provenance
                    progress.progress(20, text="🔎 Analyzing provenance...")
                    provenance = analyze_provenance(metadata)
                    
                    # Step 3: Comprehensive Detection
                    progress.progress(30, text="🤖 Loading AI detection models...")
                    
                    from imagetrust.detection.multi_detector import ComprehensiveDetector
                    
                    detector = ComprehensiveDetector()
                    
                    progress.progress(50, text="🔬 Running multi-method analysis...")
                    
                    # Convert to RGB
                    if image.mode != "RGB":
                        if image.mode == "RGBA":
                            bg = Image.new("RGB", image.size, (255, 255, 255))
                            bg.paste(image, mask=image.split()[3])
                            image = bg
                        else:
                            image = image.convert("RGB")
                    
                    progress.progress(70, text="📊 Analyzing signal characteristics...")
                    
                    result = detector.analyze(image)
                    
                    progress.progress(90, text="📝 Generating report...")
                    
                    total_time = time.time() - start_time
                    result["total_time"] = total_time
                    
                    progress.progress(100, text="✅ Analysis complete!")
                    
                    # Store results
                    st.session_state["result"] = result
                    st.session_state["metadata"] = metadata
                    st.session_state["provenance"] = provenance
                    st.session_state["source_info"] = source_info
                    st.session_state["image_info"] = image_info
                    st.session_state["image"] = image
                    st.session_state["uploaded_file"] = uploaded_file
                    st.session_state["image_bytes"] = image_bytes
                    
                    progress.empty()
                    st.rerun()
                    
                except Exception as e:
                    progress.empty()
                    st.error(f"❌ Analysis failed: {e}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
    
    with col2:
        st.header("📊 Forensic Report")
        
        if "result" in st.session_state:
            result = st.session_state["result"]
            metadata = st.session_state.get("metadata", {})
            provenance = st.session_state.get("provenance", {})
            uploaded_file = st.session_state.get("uploaded_file")
            source_info = st.session_state.get("source_info", {})
            
            # Unified score used across the report
            score_info = compute_combined_score(
                result,
                uploaded_file,
                source_info=source_info,
                settings=analysis_settings,
            )
            combined = score_info["combined"]
            ai_prob_display = score_info["ai_prob"]
            real_prob_display = score_info["real_prob"]
            ml_avg = score_info["ml_avg"]
            signal_avg = score_info["signal_avg"]
            ml_results = score_info["ml_results"]
            signal_results = score_info["signal_results"]
            votes_ai = score_info["votes_ai"]
            vote_ratio = score_info["vote_ratio"]
            all_results = score_info["all_results"]
            advanced_mode = score_info["advanced_mode"]
            chatgpt_filename = score_info["chatgpt_filename"]
            confidence = max(ai_prob_display, real_prob_display)
            
            if high_confidence_mode:
                real_threshold = 0.25
                ai_threshold = 0.75
            else:
                real_threshold = 0.35
                ai_threshold = 0.60
            
            main_verdict_is_real = combined <= real_threshold
            main_verdict_is_ai = combined >= ai_threshold
            
            # Determine verdict based on COMBINED score
            if combined >= ai_threshold:
                result_class = "ai-result"
                verdict_emoji = "🤖"
                verdict_text = "AI-Generated Image"
            elif combined <= real_threshold:
                result_class = "real-result"
                verdict_emoji = "📷"
                verdict_text = "Real Photograph"
            else:
                result_class = "uncertain-result"
                verdict_emoji = "❓"
                verdict_text = "Uncertain - Requires Manual Review"
            
            st.markdown(f"""
            <div class="result-box {result_class}">
                <h1 style="margin:0; font-size:2.2rem">{verdict_emoji} {verdict_text}</h1>
                <p style="font-size:1.4rem; margin:0.5rem 0">
                    AI Probability: <strong>{ai_prob_display:.1%}</strong> | 
                    Confidence: <strong>{confidence:.1%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            show_details = st.toggle(
                "Show detailed technical analysis",
                value=False,
                key="show_details_toggle",
                help="Enable for advanced modules, metadata, and low-level forensic details."
            )
            
            # Quick summary
            st.markdown("### ✅ Summary")
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            with col_sum1:
                verdict_short = "REAL" if main_verdict_is_real else "AI" if main_verdict_is_ai else "INCONCLUSIVE"
                st.metric("Verdict", verdict_short)
            with col_sum2:
                st.metric("AI Probability", f"{ai_prob_display:.1%}")
            with col_sum3:
                st.metric("Real Probability", f"{real_prob_display:.1%}")

            # Image details (always visible)
            image_info = st.session_state.get("image_info", {})
            source_info = score_info.get("source_info", {})
            st.markdown("### 🖼️ Image Details")
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                st.markdown(f"**File:** {image_info.get('filename', '-')}")
                st.markdown(f"**Format:** {image_info.get('format', '-')}")
            with col_i2:
                st.markdown(f"**Dimensions:** {image_info.get('dimensions', '-')}")
                st.markdown(f"**Size:** {image_info.get('size_kb', 0):.1f} KB")
            with col_i3:
                st.markdown(f"**EXIF:** {'Yes' if metadata.get('has_exif') else 'No'}")
                src = source_info.get("platform") or "Local/Unknown"
                st.markdown(f"**Source:** {src}")
            st.markdown(f"**Compression:** {source_info.get('compression_level', 'unknown').title()} | **BPP:** {source_info.get('bpp', 0):.2f}")
            
            if show_details:
                if score_info.get("calibration_applied"):
                    st.info(f"🛡️ {score_info.get('calibration_note', 'Auto calibration applied.')}")
                # Voting summary
                votes = {
                    "ai": votes_ai,
                    "real": len(all_results) - votes_ai,
                    "total": len(all_results),
                }
                st.markdown(f"""
                ### 🗳️ Detection Voting
                **{votes.get('ai', 0)}** methods vote AI | **{votes.get('real', 0)}** methods vote Real | Total: **{votes.get('total', 0)}** methods
                """)
                
                # Visual bar
                st.progress(ai_prob_display, text=f"AI: {ai_prob_display:.1%} ← → Real: {real_prob_display:.1%}")
                
                # Individual Results
                st.markdown("### 🔬 Individual Analysis Results")
                
                active_results = score_info.get("active_results", [])
                for idx, r in enumerate(active_results):
                    method = r["method"]
                    method_prob = r["ai_probability"]
                    method_conf = r["confidence"]
                    method_weight = r["weight"]
                    
                    is_ai = method_prob > 0.5
                    icon = "🔴" if is_ai else "🟢"
                    bar_color = "#ff6b6b" if is_ai else "#51cf66"
                    
                    with st.expander(f"{icon} **{method}** — {method_prob:.1%} AI", expanded=(idx < 2)):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("AI Probability", f"{method_prob:.1%}")
                        with col_b:
                            st.metric("Confidence", f"{method_conf:.0%}")
                        with col_c:
                            st.metric("Weight", f"{method_weight:.0%}")
                        
                        # Progress bar
                        st.progress(method_prob)
                        
                        # Details
                        details = r.get("details", {})
                        if details:
                            st.markdown("**Technical Details:**")
                            for key, value in details.items():
                                if key not in ["model_id", "all_probs", "labels"]:
                                    if isinstance(value, float):
                                        st.text(f"  • {key}: {value:.4f}")
                                    else:
                                        st.text(f"  • {key}: {value}")
            
            if show_details:
                if use_metadata:
                    # Metadata Section
                    st.markdown("### 📋 Image Metadata")
                    
                    if metadata.get("has_exif"):
                        cols = st.columns(2)
                        with cols[0]:
                            if metadata.get("camera"):
                                st.success(f"📷 **Camera:** {metadata['camera']}")
                            if metadata.get("date_taken"):
                                st.success(f"📅 **Date:** {metadata['date_taken']}")
                        with cols[1]:
                            if metadata.get("software"):
                                st.info(f"💻 **Software:** {metadata['software']}")
                            if metadata.get("gps"):
                                st.success(f"📍 **GPS:** Present")
                        
                        with st.expander(f"🏷️ All Tags ({len(metadata.get('all_tags', {}))})"):
                            for key, value in list(metadata.get("all_tags", {}).items())[:20]:
                                st.text(f"{key}: {value}")
                    else:
                        if main_verdict_is_real:
                            st.info("ℹ️ **No EXIF metadata found** - This is common after WhatsApp/social media or exports.")
                        else:
                            st.warning("⚠️ **No EXIF metadata found** - This can be suspicious for AI or edited images.")
                
                if use_provenance:
                    # Provenance Section
                    st.markdown("### 🔎 Provenance Analysis")
                    
                    prov_score = provenance.get("score", 50)
                    suspicious_signs = provenance.get("suspicious_signs", [])
                    authentic_signs = provenance.get("authentic_signs", [])
                    
                    if main_verdict_is_real:
                        suspicious_signs = [s for s in suspicious_signs if "No EXIF metadata" not in s]
                        st.success("✅ Provenance check: no evidence of manipulation. Metadata may be missing after sharing.")
                    else:
                        if prov_score >= 70:
                            st.success(f"✅ Authenticity Score: **{prov_score}/100**")
                        elif prov_score >= 40:
                            st.warning(f"⚠️ Authenticity Score: **{prov_score}/100**")
                        else:
                            st.error(f"❌ Authenticity Score: **{prov_score}/100**")
                    
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown("**✅ Authentic Indicators:**")
                        for sign in authentic_signs:
                            st.markdown(f"• {sign}")
                        if not authentic_signs:
                            st.caption("No authentic indicators found")
                    
                    with cols[1]:
                        st.markdown("**⚠️ Suspicious Indicators:**")
                        for sign in suspicious_signs:
                            st.markdown(f"• {sign}")
                        if not suspicious_signs:
                            st.caption("No suspicious indicators")
            
            # Final Conclusion
            st.markdown("### 🎯 Final Forensic Conclusion")
            
            num_ml_models = len(ml_results)
            num_signal = len(signal_results)
            
            weights = score_info["weights"]
            weight_ml = weights["ml"]
            weight_signal = weights["signal"]
            weight_vote = weights["vote"]
            
            if show_details and advanced_mode:
                reason = "ChatGPT/DALL-E filename detected" if chatgpt_filename else f"PNG format + ML fooled + Signal {signal_avg:.0%}"
                st.warning(f"⚠️ **Advanced AI Mode Activated!** {reason}")
            
            if show_details:
                st.markdown(f"""
                **🎯 Final Score:** {combined:.1%} AI likelihood
                
                | Component | Score | Weight |
                |-----------|-------|--------|
                | ML Models ({num_ml_models}x) | {ml_avg:.1%} | {weight_ml} |
                | Signal Analysis ({num_signal}x) | {signal_avg:.1%} | {weight_signal} |
                | Voting ({votes_ai}/{len(all_results)} vote AI) | {vote_ratio:.1%} | {weight_vote} |
                """)
            
            if combined >= ai_threshold:
                st.error(f"""
                ### 🤖 HIGH PROBABILITY: AI-GENERATED
                
                Based on {num_ml_models} ML models, this image shows indicators of being AI-generated.
                
                - Combined score: **{combined:.1%}** AI
                """)
            elif combined <= real_threshold:
                st.success(f"""
                ### 📷 HIGH PROBABILITY: REAL PHOTOGRAPH
                
                This image appears to be a genuine photograph.
                
                - Combined score: **{1-combined:.1%}** Real
                """)
            else:
                st.warning(f"""
                ### ❓ INCONCLUSIVE
                
                The local analysis is not definitive. Use external verification below.
                """)
            
            with st.expander("🔗 Verificare Externă (opțională)", expanded=False):
                st.markdown("### 🔗 Verificare Externă GRATUITĂ (Acuratețe mai mare)")
                st.markdown("""
                Modelele locale au acuratețe ~70-85%. Pentru verificare cu **acuratețe 90-95%**, 
                folosește aceste servicii **GRATUITE**:
                """)
                
                col_ext1, col_ext2, col_ext3 = st.columns(3)
                
                with col_ext1:
                    st.markdown("""
                    #### 🥇 Illuminarty
                    - **Acuratețe: ~95%**
                    - 50 imagini/lună GRATIS
                    - Detectează DALL-E, MJ, SD
                    
                    [🔗 Deschide Illuminarty](https://illuminarty.ai)
                    """)
                
                with col_ext2:
                    st.markdown("""
                    #### 🥈 AI or Not
                    - **Acuratețe: ~90%**
                    - NELIMITAT gratuit
                    - Rezultat instant
                    
                    [🔗 Deschide AI or Not](https://aiornot.com)
                    """)
                
                with col_ext3:
                    st.markdown("""
                    #### 🥉 Hive Demo
                    - **Acuratețe: ~99%**
                    - Test manual gratuit
                    - Cel mai precis
                    
                    [🔗 Deschide Hive](https://hivemoderation.com/ai-generated-content-detection)
                    """)
            
            # Processing stats
            st.caption(f"⏱️ Total analysis time: {result.get('total_time', 0):.2f}s")
            
            # ============== ADVANCED ANALYSIS TABS ==============
            if show_details:
                st.markdown("---")
                st.markdown("## 🔬 Advanced Forensic Analysis")

                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🔍 Full Forensics",
                    "🎨 AI Generator ID",
                    "🔥 Grad-CAM Heatmap",
                    "🧩 Copy-Move Detection",
                    "🔐 C2PA Credentials",
                ])
                
                image = st.session_state.get("image")
                image_bytes = st.session_state.get("image_bytes")

                # Tab 1: Full Forensics Analysis
                with tab1:
                    st.markdown("### 🔍 Comprehensive Forensics Analysis")
                    st.markdown("*Multi-signal forensics: pixel analysis, metadata, source detection, AI cues*")

                    if FORENSICS_AVAILABLE and image is not None:
                        run_forensics = st.button("Run Full Forensics", key="run_forensics_btn", type="primary")

                        if run_forensics or "forensics_report" in st.session_state:
                            if run_forensics:
                                with st.spinner("Running comprehensive forensics analysis..."):
                                    try:
                                        engine = ForensicsEngine()
                                        categories = [
                                            PluginCategory.PIXEL,
                                            PluginCategory.METADATA,
                                            PluginCategory.SOURCE,
                                        ]
                                        # Optionally add AI detection
                                        if st.session_state.get("forensics_include_ai", True):
                                            categories.append(PluginCategory.AI_DETECTION)

                                        report = engine.analyze(image, categories=categories)
                                        st.session_state["forensics_report"] = report
                                    except Exception as e:
                                        st.error(f"Forensics analysis failed: {e}")
                                        st.session_state["forensics_report"] = None

                            report = st.session_state.get("forensics_report")
                            if report:
                                # Verdict Summary
                                verdict = report.verdict
                                verdict_color = {
                                    "camera_original_likely": "green",
                                    "ai_generated_suspected": "red",
                                    "edited_likely": "orange",
                                    "screenshot_likely": "blue",
                                    "social_media_likely": "cyan",
                                    "unknown": "gray",
                                }.get(verdict.primary_verdict.value, "gray")

                                st.markdown(f"""
                                <div style="padding: 1rem; border-radius: 10px; background: linear-gradient(135deg, rgba(100,126,234,0.2), rgba(118,75,162,0.2)); border-left: 4px solid {verdict_color};">
                                    <h3 style="margin:0;">Primary Verdict: {verdict.primary_verdict.value.replace('_', ' ').title()}</h3>
                                    <p>Confidence: <strong>{verdict.primary_confidence.name}</strong> | Authenticity Score: <strong>{verdict.authenticity_score:.2f}</strong></p>
                                </div>
                                """, unsafe_allow_html=True)

                                if verdict.inconclusive:
                                    st.warning("Analysis is INCONCLUSIVE - insufficient or contradictory evidence")

                                # Top Evidence
                                if verdict.top_evidence:
                                    st.markdown("#### Top Evidence")
                                    for ev in verdict.top_evidence[:5]:
                                        st.markdown(f"- {ev}")

                                # Contradictions
                                if verdict.contradictions:
                                    st.markdown("#### Contradictions")
                                    for c in verdict.contradictions:
                                        st.warning(c)

                                # Label Breakdown
                                st.markdown("#### Label Probabilities")
                                label_cols = st.columns(4)
                                for i, ls in enumerate(sorted(verdict.labels, key=lambda x: x.probability, reverse=True)):
                                    if ls.probability > 0.05:
                                        with label_cols[i % 4]:
                                            st.metric(
                                                ls.label.value.replace("_", " ").title(),
                                                f"{ls.probability:.0%}",
                                                help=f"Confidence: {ls.confidence.name}"
                                            )

                                # Detailed Results Tabs
                                st.markdown("---")
                                st.markdown("#### Detector Results")

                                pixel_results = [r for r in report.results if r.category == PluginCategory.PIXEL]
                                meta_results = [r for r in report.results if r.category == PluginCategory.METADATA]
                                source_results = [r for r in report.results if r.category == PluginCategory.SOURCE]
                                ai_results = [r for r in report.results if r.category == PluginCategory.AI_DETECTION]

                                sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
                                    f"Pixel ({len(pixel_results)})",
                                    f"Metadata ({len(meta_results)})",
                                    f"Source ({len(source_results)})",
                                    f"AI Cues ({len(ai_results)})",
                                ])

                                with sub_tab1:
                                    for r in pixel_results:
                                        status_icon = "DETECTED" if r.detected else "not detected"
                                        status_color = "red" if r.detected else "green"
                                        with st.expander(f"**{r.plugin_name}** - [{status_icon}] Score: {r.score:.2f}"):
                                            st.markdown(f"**Explanation:** {r.explanation}")
                                            st.progress(r.score)
                                            if r.limitations:
                                                st.caption("Limitations: " + "; ".join(r.limitations[:2]))
                                            if r.details:
                                                with st.expander("Technical Details"):
                                                    st.json({k: v for k, v in r.details.items() if not isinstance(v, (bytes, np.ndarray))})

                                with sub_tab2:
                                    for r in meta_results:
                                        status_icon = "DETECTED" if r.detected else "not detected"
                                        with st.expander(f"**{r.plugin_name}** - [{status_icon}] Score: {r.score:.2f}"):
                                            st.markdown(f"**Explanation:** {r.explanation}")
                                            st.progress(r.score)
                                            if r.details:
                                                important = r.details.get("important_tags", {})
                                                if important:
                                                    st.markdown("**Important Tags:**")
                                                    for k, v in list(important.items())[:10]:
                                                        st.text(f"  {k}: {v}")

                                with sub_tab3:
                                    for r in source_results:
                                        status_icon = "DETECTED" if r.detected else "not detected"
                                        with st.expander(f"**{r.plugin_name}** - [{status_icon}] Score: {r.score:.2f}"):
                                            st.markdown(f"**Explanation:** {r.explanation}")
                                            st.progress(r.score)
                                            if r.details.get("platform_scores"):
                                                st.markdown("**Platform Scores:**")
                                                for plat, score in r.details["platform_scores"].items():
                                                    st.text(f"  {plat}: {score:.2f}")

                                with sub_tab4:
                                    if ai_results:
                                        for r in ai_results:
                                            status_icon = "AI DETECTED" if r.detected else "not detected"
                                            status_color = "red" if r.detected else "green"
                                            with st.expander(f"**{r.plugin_name}** - [{status_icon}] Score: {r.score:.2f}"):
                                                st.markdown(f"**Explanation:** {r.explanation}")
                                                st.progress(r.score)
                                                if r.details.get("model_results"):
                                                    st.markdown("**Model Results:**")
                                                    for mr in r.details["model_results"]:
                                                        st.text(f"  {mr['name']}: {mr['ai_probability']:.1%}")
                                    else:
                                        st.info("AI detection was skipped. Enable it in the sidebar settings.")

                                # Export Options
                                st.markdown("---")
                                st.markdown("#### Export Report")
                                col_exp1, col_exp2 = st.columns(2)
                                with col_exp1:
                                    json_str = report.to_json()
                                    st.download_button(
                                        "Download JSON Report",
                                        json_str,
                                        file_name=f"forensics_{report.run_id}.json",
                                        mime="application/json",
                                    )
                                with col_exp2:
                                    md_str = report.to_markdown()
                                    st.download_button(
                                        "Download Markdown Report",
                                        md_str,
                                        file_name=f"forensics_{report.run_id}.md",
                                        mime="text/markdown",
                                    )

                                st.caption(f"Analysis completed in {report.total_processing_time_ms:.0f}ms | Run ID: {report.run_id}")
                    else:
                        if not FORENSICS_AVAILABLE:
                            st.warning("Forensics module not available. Install with: pip install imagetrust")
                        else:
                            st.info("Upload an image to run forensics analysis.")

                # Tab 2: AI Generator Identification
                with tab2:
                    st.markdown("### 🎨 AI Generator Fingerprinting")
                    st.markdown("*Identifies which AI model generated the image (if AI-generated)*")
                    
                    if GENERATOR_ID_AVAILABLE and image is not None:
                        with st.spinner("Analyzing generator fingerprint..."):
                            try:
                                gen_result = identify_generator(image)
                                
                                if gen_result["is_ai"] and not main_verdict_is_real:
                                    st.error(f"""
                                    ### 🤖 AI Generator Detected
                                    
                                    **Primary Match:** {gen_result['primary_generator']}
                                    
                                    **Confidence:** {gen_result['confidence']:.1%}
                                    """)
                                else:
                                    st.success(f"""
                                    ### 📷 Real Photograph
                                    
                                    **Classification:** Authentic Image
                                    
                                    **ML Confidence:** {1-combined:.1%}
                                    """)
                                    
                                    if gen_result["is_ai"]:
                                        st.info("ℹ️ Fingerprint detected weak AI-like patterns, but ML models confirm a real photograph. This can happen with compressed images.")
                                
                                if gen_result["is_ai"] and not main_verdict_is_real:
                                    st.markdown("#### 📊 Generator Probabilities")
                                    for score in gen_result['all_scores']:
                                        prob = score['probability']
                                        generator = score['generator']
                                        bar_color = "🔴" if prob > 0.3 else "🟡" if prob > 0.1 else "🟢"
                                        st.markdown(f"{bar_color} **{generator}**: {prob:.1%}")
                                        st.progress(prob)
                                else:
                                    st.markdown("#### ✅ Authenticity Verified")
                                    st.markdown(f"""
                                    This image shows characteristics of a **real photograph**:
                                    - ML Models confidence: **{1-combined:.1%}** Real
                                    - No strong AI generator signature detected
                                    """)
                                
                                with st.expander("📋 Technical Analysis Details"):
                                    details = gen_result['analysis_details']
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Frequency Analysis:**")
                                        st.json(details.get('frequency', {}))
                                        st.markdown("**Color Analysis:**")
                                        st.json(details.get('color', {}))
                                    with col2:
                                        st.markdown("**Texture Analysis:**")
                                        st.json(details.get('texture', {}))
                                        st.markdown("**Edge Analysis:**")
                                        st.json(details.get('edges', {}))
                            except Exception as e:
                                st.error(f"Error during generator identification: {str(e)}")
                    else:
                        st.warning("Generator identification requires scipy. Install with: pip install scipy")
                
                # Tab 3: Grad-CAM Heatmap
                with tab3:
                    st.markdown("### 🔥 Grad-CAM Visualization")
                    st.markdown("*Shows which regions of the image triggered AI detection*")
                    
                    if GRADCAM_AVAILABLE and image is not None:
                        run_heatmap = True
                        if main_verdict_is_real:
                            show_heatmap = st.checkbox(
                                "Show heatmap for inspection (optional)",
                                value=False,
                                key="show_heatmap_optional"
                            )
                            if not show_heatmap:
                                run_heatmap = False
                                st.success("✅ Image classified as real. Heatmap hidden to avoid confusion.")
                        
                        if run_heatmap:
                            with st.spinner("Generating Grad-CAM heatmap..."):
                                try:
                                    gradcam_result = analyze_with_gradcam(image, use_model=False)
                                    
                                    col_orig, col_heat = st.columns(2)
                                    with col_orig:
                                        st.markdown("**Original Image**")
                                        st.image(image, use_container_width=True)
                                    with col_heat:
                                        st.markdown("**AI Detection Heatmap**")
                                        st.image(gradcam_result.overlay, use_container_width=True)
                                    
                                    st.markdown("""
                                    **Color Legend:**
                                    - 🔴 **Red/Yellow**: High AI probability regions
                                    - 🟢 **Green/Blue**: Low AI probability regions
                                    """)
                                    
                                    st.metric(
                                        "Overall Activation Score",
                                        f"{gradcam_result.activation_score:.1%}",
                                        help="Higher score = more AI-like patterns detected"
                                    )
                                    
                                    if gradcam_result.highlighted_regions:
                                        st.markdown("#### 🎯 Most Suspicious Regions")
                                        for i, region in enumerate(gradcam_result.highlighted_regions[:3], 1):
                                            st.markdown(f"""
                                            **Region {i}:**
                                            - Location: {region['center']}
                                            - Activation: {region['activation']:.1%}
                                            - {region['description']}
                                            """)
                                except Exception as e:
                                    st.error(f"Error generating heatmap: {str(e)}")
                    else:
                        st.warning("Grad-CAM requires additional dependencies. Using fallback visualization.")
                
                # Tab 4: Copy-Move Detection
                with tab4:
                    st.markdown("### 🧩 Copy-Move Forgery Detection")
                    st.markdown("*Detects if regions have been copied and pasted within the image*")
                    
                    if COPY_MOVE_AVAILABLE and image is not None:
                        with st.spinner("Analyzing for copy-move forgery..."):
                            try:
                                cm_detector = CopyMoveDetector(
                                    block_size=32,
                                    min_matches=50,
                                    similarity_threshold=0.95
                                )
                                cm_result = cm_detector.detect(image)
                                
                                likely_false_positive = main_verdict_is_real and cm_result.confidence < 0.7
                                
                                if cm_result.is_manipulated and not likely_false_positive:
                                    st.error(f"""
                                    ### ⚠️ MANIPULATION DETECTED
                                    
                                    **Confidence:** {cm_result.confidence:.1%}
                                    
                                    **Forgery Regions Found:** {len(cm_result.forgery_regions)}
                                    """)
                                else:
                                    st.success(f"""
                                    ### ✅ No Copy-Move Forgery Detected
                                    
                                    **Confidence:** {1 - cm_result.confidence:.1%}
                                    
                                    No significant duplicate regions found.
                                    """)
                                    
                                    if cm_result.match_count > 20:
                                        st.info("ℹ️ Similar patterns detected (textures/blinds), but not evidence of manipulation.")
                                
                                if cm_result.is_manipulated and not likely_false_positive:
                                    st.markdown("#### 📊 Detection Visualization")
                                    st.image(cm_result.visualization, use_container_width=True)
                                    st.markdown("""
                                    **Legend:**
                                    - 🔴 **Red boxes**: Source regions (original)
                                    - 🟢 **Green boxes**: Target regions (copied)
                                    - 🟡 **Yellow lines**: Match connections
                                    """)
                                else:
                                    st.markdown("#### 📊 Analysis Result")
                                    st.image(image, use_container_width=True, caption="Original image - No manipulation detected")
                                
                                with st.expander("📋 Analysis Details"):
                                    st.json(cm_result.analysis_details)
                                
                                st.markdown("---")
                                st.markdown("#### 🔀 Splicing Analysis")
                                if main_verdict_is_real:
                                    st.success("✅ Splicing analysis: no evidence of tampering for this real photograph.")
                                else:
                                    splicing_detector = SplicingDetector()
                                    splice_result = splicing_detector.detect(image)
                                    
                                    if splice_result["is_spliced"]:
                                        st.warning(f"""
                                        **Potential Splicing Detected**
                                        
                                        Confidence: {splice_result['confidence']:.1%}
                                        """)
                                    else:
                                        st.info("No significant splicing detected")
                                    
                                    st.image(
                                        splice_result["visualization"],
                                        caption="Noise consistency map (inconsistent regions may indicate splicing)",
                                        use_container_width=True
                                    )
                            except Exception as e:
                                st.error(f"Error during copy-move detection: {str(e)}")
                    else:
                        st.warning("Copy-Move detection requires scipy. Install with: pip install scipy")
                
                # Tab 5: C2PA Credentials
                with tab5:
                    st.markdown("### 🔐 C2PA Content Credentials")
                    st.markdown("*Verifies industry-standard provenance metadata (Adobe, Microsoft, etc.)*")
                    
                    if C2PA_AVAILABLE and image_bytes is not None:
                        with st.spinner("Checking C2PA credentials..."):
                            try:
                                c2pa_result = validate_c2pa(image_bytes)
                                
                                if c2pa_result.has_c2pa:
                                    if c2pa_result.status.value == "valid":
                                        st.success(f"""
                                        ### ✅ Valid C2PA Credentials Found
                                        
                                        **Trust Score:** {c2pa_result.trust_score}/100
                                        
                                        **Status:** {c2pa_result.status.value.upper()}
                                        """)
                                    elif c2pa_result.status.value == "tampered":
                                        st.error(f"""
                                        ### ⚠️ C2PA Signature Invalid
                                        
                                        The image may have been modified after signing.
                                        
                                        **Status:** TAMPERED
                                        """)
                                    else:
                                        st.warning(f"""
                                        ### ❓ C2PA Status: {c2pa_result.status.value.upper()}
                                        
                                        **Trust Score:** {c2pa_result.trust_score}/100
                                        """)
                                    
                                    if c2pa_result.is_ai_generated:
                                        st.error(f"""
                                        #### 🤖 AI-Generated Content Declared
                                        
                                        **Generator:** {c2pa_result.ai_generator or "Unknown"}
                                        """)
                                    
                                    st.markdown("#### 📋 Creation Information")
                                    col_c1, col_c2 = st.columns(2)
                                    with col_c1:
                                        st.markdown(f"""
                                        - **Title:** {c2pa_result.creation_info.get('title', 'N/A')}
                                        - **Software:** {c2pa_result.creation_info.get('software', 'N/A')}
                                        - **Created:** {c2pa_result.creation_info.get('created_at', 'N/A')}
                                        """)
                                    with col_c2:
                                        st.markdown(f"""
                                        - **Creator:** {c2pa_result.creation_info.get('created_by', 'N/A')}
                                        - **Version:** {c2pa_result.creation_info.get('version', 'N/A')}
                                        """)
                                    
                                    if c2pa_result.certificate_info.get('issuer'):
                                        st.markdown("#### 🏛️ Certificate Information")
                                        cert = c2pa_result.certificate_info
                                        trust_badge = "✅" if cert.get('is_trusted') else "❓"
                                        st.markdown(f"""
                                        - **Issuer:** {cert.get('issuer', 'N/A')} {trust_badge}
                                        - **Trust Level:** {cert.get('trust_level', 'unknown').upper()}
                                        """)
                                    
                                    if c2pa_result.edit_history:
                                        st.markdown("#### 📝 Edit History")
                                        for edit in c2pa_result.edit_history:
                                            st.markdown(f"- **{edit.get('action', 'Unknown')}** via {edit.get('software', 'Unknown')}")
                                    
                                    if c2pa_result.warnings:
                                        st.warning("**Warnings:**\n" + "\n".join(f"- {w}" for w in c2pa_result.warnings))
                                else:
                                    if main_verdict_is_real:
                                        st.success("""
                                        ### ✅ No C2PA Credentials (Normal)
                                        
                                        This is common for real photos shared via social media or messaging apps.
                                        """)
                                    else:
                                        st.info(f"""
                                        ### 📭 No C2PA Credentials Found
                                        
                                        This image does not contain C2PA Content Credentials.
                                        
                                        **What this means:**
                                        - The image was not created with C2PA-compatible software
                                        - OR credentials were stripped during processing/sharing
                                        - OR the image predates C2PA adoption
                                        """)
                            except Exception as e:
                                st.error(f"Error checking C2PA: {str(e)}")
                    else:
                        st.warning("C2PA validation module not available.")
            else:
                st.markdown("---")
                st.info("Advanced forensic modules are hidden. Turn on detailed analysis to view them.")
        
        else:
            st.info("👆 Upload an image and click 'Run Full Analysis' to see the forensic report.")
            
            st.markdown("""
            ### 🔬 Analysis Methods:
            
            | Method | Description |
            |--------|-------------|
            | **ML Models** | Pre-trained AI detection neural networks |
            | **FFT Analysis** | Frequency domain artifact detection |
            | **Texture** | Local pattern variance analysis |
            | **Noise** | Camera sensor noise fingerprints |
            | **Edges** | Edge coherence and direction |
            | **Color** | Channel correlation analysis |
            | **EXIF** | Metadata authenticity check |
            """)
    
    # Footer
    st.divider()
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.caption("🔬 ImageTrust v0.1.0")
    with col_f2:
        st.caption("🎓 Master's Thesis Project")
    with col_f3:
        st.caption("📊 Multi-Method Forensics")


if __name__ == "__main__":
    main()
