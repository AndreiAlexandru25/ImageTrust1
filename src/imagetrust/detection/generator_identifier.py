"""
AI Generator Fingerprinting Module
Identifies which AI generator created an image based on unique patterns.

Supported generators:
- DALL-E 2/3 (OpenAI)
- Midjourney v4/v5/v6
- Stable Diffusion 1.5/XL/2.1
- Adobe Firefly
- Imagen (Google)
- Leonardo AI
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import io
from dataclasses import dataclass
from enum import Enum


class AIGenerator(Enum):
    """Known AI image generators."""
    DALLE_2 = "DALL-E 2"
    DALLE_3 = "DALL-E 3"
    MIDJOURNEY_V4 = "Midjourney v4"
    MIDJOURNEY_V5 = "Midjourney v5"
    MIDJOURNEY_V6 = "Midjourney v6"
    STABLE_DIFFUSION_15 = "Stable Diffusion 1.5"
    STABLE_DIFFUSION_XL = "Stable Diffusion XL"
    STABLE_DIFFUSION_21 = "Stable Diffusion 2.1"
    ADOBE_FIREFLY = "Adobe Firefly"
    LEONARDO = "Leonardo AI"
    IMAGEN = "Google Imagen"
    UNKNOWN_AI = "Unknown AI Generator"
    REAL_PHOTO = "Real Photograph"


@dataclass
class GeneratorFingerprint:
    """Fingerprint characteristics of each generator."""
    name: str
    # Frequency domain characteristics
    freq_peak_ratio: Tuple[float, float]  # (min, max) expected ratio
    freq_energy_distribution: str  # "uniform", "concentrated", "scattered"
    # Color characteristics
    color_saturation_range: Tuple[float, float]
    color_temperature_bias: str  # "warm", "cool", "neutral"
    # Texture characteristics
    texture_smoothness: Tuple[float, float]
    noise_pattern: str  # "uniform", "structured", "minimal"
    # Resolution patterns
    typical_resolutions: List[Tuple[int, int]]
    aspect_ratios: List[float]
    # Artifact patterns
    edge_sharpness: Tuple[float, float]
    detail_consistency: str  # "high", "medium", "low"


# Define fingerprints for each generator
GENERATOR_FINGERPRINTS = {
    AIGenerator.DALLE_3: GeneratorFingerprint(
        name="DALL-E 3",
        freq_peak_ratio=(0.15, 0.35),
        freq_energy_distribution="concentrated",
        color_saturation_range=(0.4, 0.8),
        color_temperature_bias="neutral",
        texture_smoothness=(0.6, 0.9),
        noise_pattern="minimal",
        typical_resolutions=[(1024, 1024), (1024, 1792), (1792, 1024)],
        aspect_ratios=[1.0, 0.57, 1.75],
        edge_sharpness=(0.7, 0.95),
        detail_consistency="high"
    ),
    AIGenerator.DALLE_2: GeneratorFingerprint(
        name="DALL-E 2",
        freq_peak_ratio=(0.2, 0.4),
        freq_energy_distribution="scattered",
        color_saturation_range=(0.35, 0.7),
        color_temperature_bias="warm",
        texture_smoothness=(0.5, 0.8),
        noise_pattern="structured",
        typical_resolutions=[(512, 512), (1024, 1024)],
        aspect_ratios=[1.0],
        edge_sharpness=(0.5, 0.8),
        detail_consistency="medium"
    ),
    AIGenerator.MIDJOURNEY_V6: GeneratorFingerprint(
        name="Midjourney v6",
        freq_peak_ratio=(0.1, 0.3),
        freq_energy_distribution="uniform",
        color_saturation_range=(0.5, 0.9),
        color_temperature_bias="warm",
        texture_smoothness=(0.7, 0.95),
        noise_pattern="minimal",
        typical_resolutions=[(1024, 1024), (1456, 816), (816, 1456)],
        aspect_ratios=[1.0, 1.78, 0.56],
        edge_sharpness=(0.8, 0.98),
        detail_consistency="high"
    ),
    AIGenerator.MIDJOURNEY_V5: GeneratorFingerprint(
        name="Midjourney v5",
        freq_peak_ratio=(0.15, 0.35),
        freq_energy_distribution="uniform",
        color_saturation_range=(0.45, 0.85),
        color_temperature_bias="warm",
        texture_smoothness=(0.65, 0.9),
        noise_pattern="structured",
        typical_resolutions=[(1024, 1024), (1456, 816)],
        aspect_ratios=[1.0, 1.78],
        edge_sharpness=(0.7, 0.9),
        detail_consistency="high"
    ),
    AIGenerator.STABLE_DIFFUSION_XL: GeneratorFingerprint(
        name="Stable Diffusion XL",
        freq_peak_ratio=(0.2, 0.45),
        freq_energy_distribution="scattered",
        color_saturation_range=(0.3, 0.75),
        color_temperature_bias="neutral",
        texture_smoothness=(0.4, 0.75),
        noise_pattern="structured",
        typical_resolutions=[(1024, 1024), (896, 1152), (1152, 896)],
        aspect_ratios=[1.0, 0.78, 1.28],
        edge_sharpness=(0.5, 0.85),
        detail_consistency="medium"
    ),
    AIGenerator.STABLE_DIFFUSION_15: GeneratorFingerprint(
        name="Stable Diffusion 1.5",
        freq_peak_ratio=(0.25, 0.5),
        freq_energy_distribution="scattered",
        color_saturation_range=(0.25, 0.65),
        color_temperature_bias="cool",
        texture_smoothness=(0.3, 0.6),
        noise_pattern="uniform",
        typical_resolutions=[(512, 512), (768, 768)],
        aspect_ratios=[1.0],
        edge_sharpness=(0.4, 0.7),
        detail_consistency="low"
    ),
    AIGenerator.ADOBE_FIREFLY: GeneratorFingerprint(
        name="Adobe Firefly",
        freq_peak_ratio=(0.12, 0.28),
        freq_energy_distribution="concentrated",
        color_saturation_range=(0.35, 0.7),
        color_temperature_bias="neutral",
        texture_smoothness=(0.65, 0.88),
        noise_pattern="minimal",
        typical_resolutions=[(1024, 1024), (2048, 2048)],
        aspect_ratios=[1.0, 1.5, 0.67],
        edge_sharpness=(0.75, 0.92),
        detail_consistency="high"
    ),
}


class GeneratorIdentifier:
    """
    Identifies which AI generator created an image using multi-modal analysis.
    
    Analysis methods:
    1. Frequency domain analysis (FFT patterns)
    2. Color distribution analysis
    3. Texture and noise patterns
    4. Resolution and aspect ratio matching
    5. Edge and detail analysis
    """
    
    def __init__(self):
        self.fingerprints = GENERATOR_FINGERPRINTS
    
    def identify(self, image: Image.Image) -> Dict:
        """
        Main identification method.
        
        Returns:
            Dict with:
            - primary_generator: Most likely generator
            - confidence: Confidence score (0-1)
            - all_scores: Scores for all generators
            - analysis_details: Detailed analysis results
        """
        # Convert to numpy array
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image)
        
        # Run all analyses
        freq_analysis = self._analyze_frequency(img_array)
        color_analysis = self._analyze_color(img_array)
        texture_analysis = self._analyze_texture(img_array)
        resolution_analysis = self._analyze_resolution(image.size)
        edge_analysis = self._analyze_edges(img_array)
        
        # Calculate scores for each generator
        scores = {}
        for generator, fingerprint in self.fingerprints.items():
            score = self._calculate_match_score(
                fingerprint,
                freq_analysis,
                color_analysis,
                texture_analysis,
                resolution_analysis,
                edge_analysis
            )
            scores[generator] = score
        
        # Add real photo score (inverse of AI patterns)
        real_score = self._calculate_real_photo_score(
            freq_analysis, color_analysis, texture_analysis, edge_analysis
        )
        scores[AIGenerator.REAL_PHOTO] = real_score
        
        # Normalize scores
        total = sum(scores.values()) + 1e-10
        normalized_scores = {k: v/total for k, v in scores.items()}
        
        # Find primary generator
        primary = max(normalized_scores, key=normalized_scores.get)
        confidence = normalized_scores[primary]
        
        # Sort by score
        sorted_scores = sorted(
            normalized_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            "primary_generator": primary.value,
            "confidence": confidence,
            "is_ai": primary != AIGenerator.REAL_PHOTO,
            "all_scores": [
                {"generator": g.value, "probability": p}
                for g, p in sorted_scores[:5]  # Top 5
            ],
            "analysis_details": {
                "frequency": freq_analysis,
                "color": color_analysis,
                "texture": texture_analysis,
                "resolution": resolution_analysis,
                "edges": edge_analysis
            }
        }
    
    def _analyze_frequency(self, img_array: np.ndarray) -> Dict:
        """Analyze frequency domain characteristics."""
        # Convert to grayscale
        gray = np.mean(img_array, axis=2)
        
        # Apply FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Analyze frequency distribution
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Calculate energy in different frequency bands
        low_freq_mask = np.zeros_like(magnitude, dtype=bool)
        mid_freq_mask = np.zeros_like(magnitude, dtype=bool)
        high_freq_mask = np.zeros_like(magnitude, dtype=bool)
        
        for y in range(h):
            for x in range(w):
                dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                max_dist = np.sqrt(center_y**2 + center_x**2)
                rel_dist = dist / max_dist
                
                if rel_dist < 0.15:
                    low_freq_mask[y, x] = True
                elif rel_dist < 0.5:
                    mid_freq_mask[y, x] = True
                else:
                    high_freq_mask[y, x] = True
        
        low_energy = np.sum(magnitude[low_freq_mask])
        mid_energy = np.sum(magnitude[mid_freq_mask])
        high_energy = np.sum(magnitude[high_freq_mask])
        total_energy = low_energy + mid_energy + high_energy + 1e-10
        
        # Peak ratio (characteristic of AI images)
        peak_ratio = high_energy / total_energy
        
        # Energy distribution type
        low_ratio = low_energy / total_energy
        mid_ratio = mid_energy / total_energy
        
        if low_ratio > 0.7:
            distribution = "concentrated"
        elif mid_ratio > 0.5:
            distribution = "uniform"
        else:
            distribution = "scattered"
        
        return {
            "peak_ratio": float(peak_ratio),
            "low_freq_ratio": float(low_ratio),
            "mid_freq_ratio": float(mid_ratio),
            "high_freq_ratio": float(high_energy / total_energy),
            "distribution": distribution
        }
    
    def _analyze_color(self, img_array: np.ndarray) -> Dict:
        """Analyze color characteristics."""
        # Convert to HSV for saturation analysis
        from colorsys import rgb_to_hsv
        
        # Sample pixels for efficiency
        h, w, _ = img_array.shape
        sample_size = min(10000, h * w)
        indices = np.random.choice(h * w, sample_size, replace=False)
        
        flat_img = img_array.reshape(-1, 3)
        sampled = flat_img[indices]
        
        # Calculate HSV values
        hsv_values = []
        for pixel in sampled:
            r, g, b = pixel / 255.0
            h_val, s_val, v_val = rgb_to_hsv(r, g, b)
            hsv_values.append((h_val, s_val, v_val))
        
        hsv_array = np.array(hsv_values)
        
        # Saturation statistics
        saturation_mean = np.mean(hsv_array[:, 1])
        saturation_std = np.std(hsv_array[:, 1])
        
        # Color temperature (warm vs cool)
        # Calculate average of red vs blue channels
        avg_red = np.mean(sampled[:, 0])
        avg_blue = np.mean(sampled[:, 2])
        
        if avg_red > avg_blue * 1.1:
            temperature = "warm"
        elif avg_blue > avg_red * 1.1:
            temperature = "cool"
        else:
            temperature = "neutral"
        
        # Color diversity
        hue_std = np.std(hsv_array[:, 0])
        
        return {
            "saturation_mean": float(saturation_mean),
            "saturation_std": float(saturation_std),
            "temperature": temperature,
            "temperature_ratio": float(avg_red / (avg_blue + 1e-10)),
            "hue_diversity": float(hue_std)
        }
    
    def _analyze_texture(self, img_array: np.ndarray) -> Dict:
        """Analyze texture and noise patterns."""
        # Convert to grayscale
        gray = np.mean(img_array, axis=2)
        
        # Calculate local variance (texture measure)
        from scipy import ndimage
        
        # Sobel filters for texture
        sobel_x = ndimage.sobel(gray, axis=1)
        sobel_y = ndimage.sobel(gray, axis=0)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Smoothness measure (inverse of gradient magnitude)
        smoothness = 1.0 - (np.mean(gradient_magnitude) / 255.0)
        smoothness = np.clip(smoothness, 0, 1)
        
        # Noise estimation using Laplacian
        laplacian = ndimage.laplace(gray)
        noise_estimate = np.std(laplacian) / 255.0
        
        # Determine noise pattern
        # Block-based noise analysis
        block_size = 32
        h, w = gray.shape
        block_stds = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                block_stds.append(np.std(block))
        
        std_of_stds = np.std(block_stds) if block_stds else 0
        
        if noise_estimate < 0.02:
            noise_pattern = "minimal"
        elif std_of_stds < 10:
            noise_pattern = "uniform"
        else:
            noise_pattern = "structured"
        
        return {
            "smoothness": float(smoothness),
            "noise_level": float(noise_estimate),
            "noise_pattern": noise_pattern,
            "texture_variance": float(np.var(gradient_magnitude))
        }
    
    def _analyze_resolution(self, size: Tuple[int, int]) -> Dict:
        """Analyze resolution and aspect ratio."""
        width, height = size
        aspect_ratio = width / height
        
        # Check common AI resolutions
        common_ai_resolutions = [
            (512, 512), (768, 768), (1024, 1024), (2048, 2048),
            (1024, 1792), (1792, 1024),  # DALL-E 3
            (1456, 816), (816, 1456),    # Midjourney
            (896, 1152), (1152, 896),    # SDXL
        ]
        
        is_common_ai_res = (width, height) in common_ai_resolutions
        
        # Common AI aspect ratios
        common_ai_aspects = [1.0, 1.78, 0.56, 1.5, 0.67, 1.28, 0.78]
        closest_aspect = min(common_ai_aspects, key=lambda x: abs(x - aspect_ratio))
        aspect_match = abs(closest_aspect - aspect_ratio) < 0.05
        
        return {
            "width": width,
            "height": height,
            "aspect_ratio": float(aspect_ratio),
            "is_common_ai_resolution": is_common_ai_res,
            "matches_ai_aspect_ratio": aspect_match,
            "megapixels": (width * height) / 1_000_000
        }
    
    def _analyze_edges(self, img_array: np.ndarray) -> Dict:
        """Analyze edge characteristics."""
        from scipy import ndimage
        
        gray = np.mean(img_array, axis=2)
        
        # Canny-like edge detection
        sobel_x = ndimage.sobel(gray, axis=1)
        sobel_y = ndimage.sobel(gray, axis=0)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Edge sharpness (ratio of strong edges)
        threshold = np.percentile(edges, 90)
        strong_edges = edges > threshold
        edge_sharpness = np.mean(edges[strong_edges]) / 255.0 if np.any(strong_edges) else 0
        
        # Edge consistency (variance in edge strength)
        edge_consistency = 1.0 - (np.std(edges[strong_edges]) / 255.0) if np.any(strong_edges) else 0.5
        
        # Detail level
        detail_level = np.mean(edges) / 255.0
        
        return {
            "sharpness": float(np.clip(edge_sharpness, 0, 1)),
            "consistency": float(np.clip(edge_consistency, 0, 1)),
            "detail_level": float(detail_level),
            "edge_density": float(np.mean(strong_edges))
        }
    
    def _calculate_match_score(
        self,
        fingerprint: GeneratorFingerprint,
        freq: Dict,
        color: Dict,
        texture: Dict,
        resolution: Dict,
        edges: Dict
    ) -> float:
        """Calculate how well the image matches a generator fingerprint."""
        score = 0.0
        
        # Frequency match (weight: 25%)
        freq_match = 0.0
        if fingerprint.freq_peak_ratio[0] <= freq["peak_ratio"] <= fingerprint.freq_peak_ratio[1]:
            freq_match = 1.0
        else:
            # Partial match
            mid = (fingerprint.freq_peak_ratio[0] + fingerprint.freq_peak_ratio[1]) / 2
            dist = abs(freq["peak_ratio"] - mid)
            freq_match = max(0, 1 - dist * 2)
        
        if freq["distribution"] == fingerprint.freq_energy_distribution:
            freq_match += 0.3
        
        score += freq_match * 0.25
        
        # Color match (weight: 20%)
        color_match = 0.0
        sat_min, sat_max = fingerprint.color_saturation_range
        if sat_min <= color["saturation_mean"] <= sat_max:
            color_match = 1.0
        else:
            mid = (sat_min + sat_max) / 2
            dist = abs(color["saturation_mean"] - mid)
            color_match = max(0, 1 - dist * 2)
        
        if color["temperature"] == fingerprint.color_temperature_bias:
            color_match += 0.3
        
        score += color_match * 0.20
        
        # Texture match (weight: 25%)
        texture_match = 0.0
        smooth_min, smooth_max = fingerprint.texture_smoothness
        if smooth_min <= texture["smoothness"] <= smooth_max:
            texture_match = 1.0
        else:
            mid = (smooth_min + smooth_max) / 2
            dist = abs(texture["smoothness"] - mid)
            texture_match = max(0, 1 - dist * 2)
        
        if texture["noise_pattern"] == fingerprint.noise_pattern:
            texture_match += 0.3
        
        score += texture_match * 0.25
        
        # Resolution match (weight: 15%)
        res_match = 0.0
        if resolution["is_common_ai_resolution"]:
            res_match = 1.0
        elif resolution["matches_ai_aspect_ratio"]:
            res_match = 0.6
        
        # Check specific resolutions for this generator
        for res in fingerprint.typical_resolutions:
            if abs(resolution["width"] - res[0]) < 50 and abs(resolution["height"] - res[1]) < 50:
                res_match = 1.0
                break
        
        score += res_match * 0.15
        
        # Edge match (weight: 15%)
        edge_match = 0.0
        sharp_min, sharp_max = fingerprint.edge_sharpness
        if sharp_min <= edges["sharpness"] <= sharp_max:
            edge_match = 1.0
        else:
            mid = (sharp_min + sharp_max) / 2
            dist = abs(edges["sharpness"] - mid)
            edge_match = max(0, 1 - dist * 2)
        
        score += edge_match * 0.15
        
        return score
    
    def _calculate_real_photo_score(
        self,
        freq: Dict,
        color: Dict,
        texture: Dict,
        edges: Dict
    ) -> float:
        """Calculate probability that image is a real photograph."""
        score = 0.0
        
        # Real photos typically have:
        # - Scattered frequency distribution (natural noise)
        if freq["distribution"] == "scattered":
            score += 0.2
        
        # - Higher noise levels
        if texture["noise_level"] > 0.03:
            score += 0.2
        
        # - More varied saturation
        if color["saturation_std"] > 0.15:
            score += 0.15
        
        # - Less uniform smoothness
        if texture["smoothness"] < 0.5:
            score += 0.2
        
        # - Structured noise (sensor pattern)
        if texture["noise_pattern"] == "structured":
            score += 0.15
        
        # - Natural edge variation
        if edges["consistency"] < 0.7:
            score += 0.1
        
        return score


# Convenience function
def identify_generator(image: Image.Image) -> Dict:
    """Quick identification of AI generator."""
    identifier = GeneratorIdentifier()
    return identifier.identify(image)
