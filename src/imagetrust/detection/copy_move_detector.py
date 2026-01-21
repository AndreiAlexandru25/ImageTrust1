"""
Copy-Move Forgery Detection Module
Detects regions that have been copied and pasted within the same image.

Techniques used:
1. Block matching with DCT (Discrete Cosine Transform)
2. Feature point matching (SIFT/ORB)
3. PatchMatch algorithm
4. Keypoint clustering

This is a forensic tool used to detect image manipulation.
"""

import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import io
import base64


@dataclass
class ForgeryRegion:
    """Detected copy-move forgery region."""
    source_bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    target_bbox: Tuple[int, int, int, int]
    similarity: float  # 0-1
    num_matches: int
    match_type: str  # "exact", "scaled", "rotated"


@dataclass
class CopyMoveResult:
    """Result of copy-move detection."""
    is_manipulated: bool
    confidence: float
    forgery_regions: List[ForgeryRegion]
    visualization: Image.Image
    match_count: int
    analysis_details: Dict


class CopyMoveDetector:
    """
    Detects copy-move forgery in images using multiple techniques.
    
    Copy-move forgery is when a region of an image is copied and pasted
    elsewhere in the same image, often to hide something or duplicate content.
    """
    
    def __init__(self, 
                 block_size: int = 16,
                 min_matches: int = 10,
                 similarity_threshold: float = 0.9):
        """
        Initialize detector.
        
        Args:
            block_size: Size of blocks for matching
            min_matches: Minimum matches to consider as forgery
            similarity_threshold: Minimum similarity for match
        """
        self.block_size = block_size
        self.min_matches = min_matches
        self.similarity_threshold = similarity_threshold
    
    def detect(self, image: Image.Image) -> CopyMoveResult:
        """
        Main detection method.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            CopyMoveResult with detection results
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        
        # Method 1: Block-based DCT matching
        dct_matches = self._dct_block_matching(img_array)
        
        # Method 2: Feature point matching (ORB)
        feature_matches = self._feature_point_matching(img_array)
        
        # Method 3: Robust matching with clustering
        robust_matches = self._robust_matching(img_array)
        
        # Combine results
        all_matches = self._combine_matches(dct_matches, feature_matches, robust_matches)
        
        # Cluster matches into forgery regions
        forgery_regions = self._cluster_matches(all_matches, img_array.shape)
        
        # Calculate confidence
        confidence = self._calculate_confidence(all_matches, forgery_regions)
        
        # Create visualization
        visualization = self._create_visualization(image, forgery_regions, all_matches)
        
        # Determine if manipulated
        is_manipulated = len(forgery_regions) > 0 and confidence > 0.5
        
        return CopyMoveResult(
            is_manipulated=is_manipulated,
            confidence=confidence,
            forgery_regions=forgery_regions,
            visualization=visualization,
            match_count=len(all_matches),
            analysis_details={
                "dct_matches": len(dct_matches),
                "feature_matches": len(feature_matches),
                "robust_matches": len(robust_matches),
                "total_regions": len(forgery_regions),
                "block_size": self.block_size,
                "methods_used": ["DCT", "ORB", "Robust Matching"]
            }
        )
    
    def _dct_block_matching(self, img_array: np.ndarray) -> List[Tuple]:
        """
        Block matching using Discrete Cosine Transform.
        Effective for detecting exact copies.
        """
        from scipy.fftpack import dct
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2).astype(np.float32)
        else:
            gray = img_array.astype(np.float32)
        
        h, w = gray.shape
        block_size = self.block_size
        
        # Extract blocks and their DCT features
        blocks = {}
        block_positions = []
        
        for y in range(0, h - block_size, block_size // 2):
            for x in range(0, w - block_size, block_size // 2):
                block = gray[y:y+block_size, x:x+block_size]
                
                # Apply DCT
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # Use top-left coefficients as feature (most energy)
                feature = dct_block[:8, :8].flatten()
                
                # Quantize for matching
                feature_key = tuple(np.round(feature, 1))
                
                if feature_key in blocks:
                    blocks[feature_key].append((x, y))
                else:
                    blocks[feature_key] = [(x, y)]
                
                block_positions.append((x, y, feature))
        
        # Find matching blocks
        matches = []
        for positions in blocks.values():
            if len(positions) >= 2:
                # Multiple blocks with same features
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        p1, p2 = positions[i], positions[j]
                        
                        # Ensure sufficient distance
                        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                        if dist > block_size * 2:
                            matches.append((p1, p2, 1.0))
        
        return matches
    
    def _feature_point_matching(self, img_array: np.ndarray) -> List[Tuple]:
        """
        Feature point matching using ORB detector.
        Good for detecting scaled/rotated copies.
        """
        try:
            import cv2
        except ImportError:
            # Fallback without OpenCV
            return self._simple_feature_matching(img_array)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=2000)
        
        # Find keypoints and descriptors
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 2:
            return []
        
        # Match features with themselves
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Find 2 best matches for each descriptor
        matches_list = bf.knnMatch(descriptors, descriptors, k=3)
        
        matches = []
        for match_group in matches_list:
            if len(match_group) >= 2:
                # Skip self-match (first one), use second best
                for m in match_group[1:]:
                    if m.distance < 50:  # Good match threshold
                        pt1 = keypoints[m.queryIdx].pt
                        pt2 = keypoints[m.trainIdx].pt
                        
                        # Ensure sufficient distance
                        dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                        if dist > self.block_size * 2:
                            similarity = 1.0 - (m.distance / 256.0)
                            matches.append((
                                (int(pt1[0]), int(pt1[1])),
                                (int(pt2[0]), int(pt2[1])),
                                similarity
                            ))
        
        return matches
    
    def _simple_feature_matching(self, img_array: np.ndarray) -> List[Tuple]:
        """Fallback feature matching without OpenCV."""
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        from scipy import ndimage
        
        # Simple corner detection
        sobel_x = ndimage.sobel(gray, axis=1)
        sobel_y = ndimage.sobel(gray, axis=0)
        
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Find local maxima as keypoints
        from scipy.ndimage import maximum_filter
        
        local_max = maximum_filter(gradient_magnitude, size=20)
        keypoints = np.where((gradient_magnitude == local_max) & (gradient_magnitude > 50))
        
        # Extract patches around keypoints
        patch_size = 16
        h, w = gray.shape
        
        patches = []
        positions = []
        
        for y, x in zip(keypoints[0], keypoints[1]):
            if y >= patch_size and y < h - patch_size and x >= patch_size and x < w - patch_size:
                patch = gray[y-patch_size:y+patch_size, x-patch_size:x+patch_size]
                patches.append(patch.flatten())
                positions.append((x, y))
        
        # Match patches
        matches = []
        if len(patches) > 1:
            patches = np.array(patches)
            
            for i in range(len(patches)):
                for j in range(i + 1, len(patches)):
                    # Normalized correlation
                    p1 = patches[i] - patches[i].mean()
                    p2 = patches[j] - patches[j].mean()
                    
                    norm1 = np.linalg.norm(p1) + 1e-10
                    norm2 = np.linalg.norm(p2) + 1e-10
                    
                    correlation = np.dot(p1, p2) / (norm1 * norm2)
                    
                    if correlation > 0.9:
                        # Ensure sufficient distance
                        dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                                      (positions[i][1] - positions[j][1])**2)
                        if dist > self.block_size * 2:
                            matches.append((positions[i], positions[j], correlation))
        
        return matches[:100]  # Limit matches
    
    def _robust_matching(self, img_array: np.ndarray) -> List[Tuple]:
        """
        Robust matching using sliding window with perceptual hashing.
        More resistant to slight modifications.
        """
        # Convert to grayscale and resize for efficiency
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        h, w = gray.shape
        
        # Use larger blocks for robustness
        block_size = self.block_size * 2
        stride = block_size // 2
        
        # Extract block hashes
        block_hashes = {}
        
        for y in range(0, h - block_size, stride):
            for x in range(0, w - block_size, stride):
                block = gray[y:y+block_size, x:x+block_size]
                
                # Compute perceptual hash (simplified)
                # Resize to 8x8, compute DCT, use sign of coefficients
                from scipy.ndimage import zoom
                small = zoom(block, 8.0 / block_size, order=1)
                
                avg = small.mean()
                hash_bits = (small > avg).flatten()
                hash_key = tuple(hash_bits)
                
                if hash_key in block_hashes:
                    block_hashes[hash_key].append((x, y))
                else:
                    block_hashes[hash_key] = [(x, y)]
        
        # Find matching blocks
        matches = []
        for positions in block_hashes.values():
            if len(positions) >= 2:
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        p1, p2 = positions[i], positions[j]
                        
                        # Ensure sufficient distance
                        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                        if dist > block_size:
                            matches.append((p1, p2, 0.95))
        
        return matches
    
    def _combine_matches(self, *match_lists) -> List[Tuple]:
        """Combine matches from different methods."""
        all_matches = []
        seen = set()
        
        for matches in match_lists:
            for m in matches:
                # Create unique key for this match
                key = (
                    min(m[0], m[1]),
                    max(m[0], m[1])
                )
                
                if key not in seen:
                    seen.add(key)
                    all_matches.append(m)
        
        return all_matches
    
    def _cluster_matches(self, matches: List[Tuple], img_shape: Tuple) -> List[ForgeryRegion]:
        """Cluster matches into forgery regions."""
        if len(matches) < self.min_matches:
            return []
        
        # Group matches by displacement vector
        displacement_groups = defaultdict(list)
        
        for m in matches:
            p1, p2, sim = m
            
            # Calculate displacement
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Quantize displacement
            dx_q = round(dx / 10) * 10
            dy_q = round(dy / 10) * 10
            
            displacement_groups[(dx_q, dy_q)].append(m)
        
        # Find significant clusters
        regions = []
        for displacement, group_matches in displacement_groups.items():
            if len(group_matches) >= self.min_matches // 2:
                # Calculate bounding boxes
                source_pts = [m[0] for m in group_matches]
                target_pts = [m[1] for m in group_matches]
                
                source_bbox = (
                    min(p[0] for p in source_pts),
                    min(p[1] for p in source_pts),
                    max(p[0] for p in source_pts) + self.block_size,
                    max(p[1] for p in source_pts) + self.block_size
                )
                
                target_bbox = (
                    min(p[0] for p in target_pts),
                    min(p[1] for p in target_pts),
                    max(p[0] for p in target_pts) + self.block_size,
                    max(p[1] for p in target_pts) + self.block_size
                )
                
                avg_similarity = np.mean([m[2] for m in group_matches])
                
                regions.append(ForgeryRegion(
                    source_bbox=source_bbox,
                    target_bbox=target_bbox,
                    similarity=avg_similarity,
                    num_matches=len(group_matches),
                    match_type="exact"
                ))
        
        return regions
    
    def _calculate_confidence(self, matches: List[Tuple], regions: List[ForgeryRegion]) -> float:
        """Calculate overall confidence of forgery detection."""
        if len(matches) == 0:
            return 0.0
        
        # Base confidence from number of matches
        match_confidence = min(len(matches) / 50.0, 1.0)
        
        # Region confidence
        if len(regions) > 0:
            avg_similarity = np.mean([r.similarity for r in regions])
            avg_matches = np.mean([r.num_matches for r in regions])
            region_confidence = avg_similarity * min(avg_matches / 20.0, 1.0)
        else:
            region_confidence = 0.0
        
        # Combined confidence
        confidence = match_confidence * 0.4 + region_confidence * 0.6
        
        return float(np.clip(confidence, 0, 1))
    
    def _create_visualization(self, 
                             image: Image.Image, 
                             regions: List[ForgeryRegion],
                             matches: List[Tuple]) -> Image.Image:
        """Create visualization of detected forgeries."""
        # Create copy of image
        vis = image.copy()
        draw = ImageDraw.Draw(vis, 'RGBA')
        
        # Draw match lines (faded)
        for m in matches[:200]:  # Limit for performance
            p1, p2, sim = m
            color = (255, 0, 0, int(sim * 100))
            draw.line([p1, p2], fill=color, width=1)
        
        # Draw forgery regions
        colors = [
            (255, 0, 0, 100),    # Red for source
            (0, 255, 0, 100),    # Green for target
            (0, 0, 255, 100),    # Blue
            (255, 255, 0, 100),  # Yellow
        ]
        
        for i, region in enumerate(regions):
            color_idx = i % len(colors)
            
            # Draw source region
            draw.rectangle(region.source_bbox, 
                          outline=(255, 0, 0, 255), 
                          fill=colors[color_idx],
                          width=3)
            
            # Draw target region
            draw.rectangle(region.target_bbox,
                          outline=(0, 255, 0, 255),
                          fill=(0, 255, 0, 100),
                          width=3)
            
            # Draw connecting line
            src_center = (
                (region.source_bbox[0] + region.source_bbox[2]) // 2,
                (region.source_bbox[1] + region.source_bbox[3]) // 2
            )
            tgt_center = (
                (region.target_bbox[0] + region.target_bbox[2]) // 2,
                (region.target_bbox[1] + region.target_bbox[3]) // 2
            )
            draw.line([src_center, tgt_center], fill=(255, 255, 0, 200), width=2)
        
        return vis


class SplicingDetector:
    """
    Detects splicing forgery (combining regions from different images).
    Uses noise level analysis and JPEG artifact analysis.
    """
    
    def detect(self, image: Image.Image) -> Dict:
        """Detect splicing in image."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        
        # Analyze noise levels across image
        noise_map = self._analyze_noise_levels(img_array)
        
        # Analyze JPEG artifacts
        jpeg_inconsistency = self._analyze_jpeg_artifacts(img_array)
        
        # Detect suspicious regions
        suspicious_regions = self._find_inconsistent_regions(noise_map)
        
        # Calculate splicing probability
        is_spliced = len(suspicious_regions) > 0 and jpeg_inconsistency > 0.3
        confidence = min(len(suspicious_regions) / 5.0, 1.0) * jpeg_inconsistency
        
        # Create visualization
        vis = self._create_noise_visualization(image, noise_map)
        
        return {
            "is_spliced": is_spliced,
            "confidence": float(confidence),
            "suspicious_regions": suspicious_regions,
            "visualization": vis,
            "noise_inconsistency": float(np.std(noise_map)),
            "jpeg_inconsistency": float(jpeg_inconsistency)
        }
    
    def _analyze_noise_levels(self, img_array: np.ndarray) -> np.ndarray:
        """Analyze noise levels across image blocks."""
        from scipy import ndimage
        
        gray = np.mean(img_array, axis=2)
        h, w = gray.shape
        
        block_size = 32
        noise_map = np.zeros((h // block_size, w // block_size))
        
        for i, y in enumerate(range(0, h - block_size, block_size)):
            for j, x in enumerate(range(0, w - block_size, block_size)):
                block = gray[y:y+block_size, x:x+block_size]
                
                # Estimate noise using Laplacian
                laplacian = ndimage.laplace(block)
                noise_level = np.std(laplacian)
                
                noise_map[i, j] = noise_level
        
        return noise_map
    
    def _analyze_jpeg_artifacts(self, img_array: np.ndarray) -> float:
        """Analyze JPEG compression artifact consistency."""
        from scipy.fftpack import dct
        
        gray = np.mean(img_array, axis=2).astype(np.float32)
        h, w = gray.shape
        
        # Analyze 8x8 blocks (JPEG standard)
        block_size = 8
        artifact_scores = []
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                
                # DCT of block
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # Check for quantization patterns
                quantization_score = np.sum(np.abs(dct_block) < 1) / 64.0
                artifact_scores.append(quantization_score)
        
        # Inconsistency is the variance in artifact patterns
        return float(np.std(artifact_scores))
    
    def _find_inconsistent_regions(self, noise_map: np.ndarray) -> List[Dict]:
        """Find regions with inconsistent noise levels."""
        mean_noise = np.mean(noise_map)
        std_noise = np.std(noise_map)
        
        # Find outlier blocks
        threshold_high = mean_noise + 2 * std_noise
        threshold_low = mean_noise - 2 * std_noise
        
        suspicious = []
        h, w = noise_map.shape
        
        for i in range(h):
            for j in range(w):
                if noise_map[i, j] > threshold_high or noise_map[i, j] < threshold_low:
                    suspicious.append({
                        "block": (j * 32, i * 32),
                        "noise_level": float(noise_map[i, j]),
                        "deviation": float(abs(noise_map[i, j] - mean_noise) / std_noise) if std_noise > 0 else 0
                    })
        
        return suspicious[:20]  # Limit results
    
    def _create_noise_visualization(self, image: Image.Image, noise_map: np.ndarray) -> Image.Image:
        """Create visualization of noise inconsistencies."""
        from scipy.ndimage import zoom
        
        # Resize noise map to image size
        zoom_h = image.height / noise_map.shape[0]
        zoom_w = image.width / noise_map.shape[1]
        noise_resized = zoom(noise_map, (zoom_h, zoom_w), order=1)
        
        # Normalize and convert to color
        noise_norm = (noise_resized - noise_resized.min()) / (noise_resized.max() - noise_resized.min() + 1e-8)
        
        import matplotlib.cm as cm
        colormap = cm.get_cmap('RdYlBu_r')  # Red = high noise, Blue = low noise
        noise_colored = colormap(noise_norm)
        noise_colored = (noise_colored[:, :, :3] * 255).astype(np.uint8)
        
        noise_img = Image.fromarray(noise_colored)
        
        # Blend with original
        overlay = Image.blend(image.convert('RGB'), noise_img, alpha=0.3)
        
        return overlay


def detect_copy_move(image: Image.Image) -> CopyMoveResult:
    """Convenience function for copy-move detection."""
    detector = CopyMoveDetector()
    return detector.detect(image)


def detect_splicing(image: Image.Image) -> Dict:
    """Convenience function for splicing detection."""
    detector = SplicingDetector()
    return detector.detect(image)
