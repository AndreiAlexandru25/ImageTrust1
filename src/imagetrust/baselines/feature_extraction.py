"""
Forensic feature extraction for classical baseline (B1).

Extracts handcrafted features commonly used in image forensics:
- DCT coefficients (frequency domain)
- Noise residuals
- JPEG artifact indicators
- Color channel statistics
- Local Binary Patterns (texture)
- Edge statistics

References:
- Fridrich & Kodovsky (2012) - Rich Models for Steganalysis
- Zhou et al. (2018) - Learning Rich Features for Image Manipulation Detection
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    # DCT
    use_dct: bool = True
    dct_coeffs: int = 64

    # Noise
    use_noise: bool = True
    noise_sigma_bins: int = 10

    # JPEG artifacts
    use_jpeg_artifacts: bool = True

    # Color
    use_color_stats: bool = True
    color_bins: int = 32

    # Texture (LBP)
    use_lbp: bool = True
    lbp_radius: int = 1
    lbp_points: int = 8

    # Edges
    use_edges: bool = True


class ForensicFeatureExtractor:
    """
    Extract forensic features from images.

    Features are designed to capture:
    1. Frequency-domain artifacts (GAN fingerprints in DCT)
    2. Noise inconsistencies (real cameras have characteristic noise)
    3. Compression artifacts (JPEG blocking)
    4. Statistical anomalies (color distributions)
    5. Texture patterns (LBP for local structure)

    Example:
        >>> extractor = ForensicFeatureExtractor()
        >>> features = extractor.extract(image)
        >>> print(f"Feature vector: {features.shape}")  # (num_features,)
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._feature_names: List[str] = []

    @property
    def feature_names(self) -> List[str]:
        """Return list of feature names for interpretability."""
        return self._feature_names

    @property
    def num_features(self) -> int:
        """Return total number of features."""
        # Compute based on config
        n = 0
        if self.config.use_dct:
            n += self.config.dct_coeffs
        if self.config.use_noise:
            n += self.config.noise_sigma_bins + 4  # bins + stats
        if self.config.use_jpeg_artifacts:
            n += 8  # blocking metrics
        if self.config.use_color_stats:
            n += self.config.color_bins * 3 + 12  # histograms + stats
        if self.config.use_lbp:
            n += 2 ** self.config.lbp_points  # LBP histogram
        if self.config.use_edges:
            n += 8  # edge statistics
        return n

    def extract(self, image: Image.Image) -> np.ndarray:
        """
        Extract all features from an image.

        Args:
            image: PIL Image (will be converted to RGB)

        Returns:
            1D numpy array of features
        """
        image = image.convert("RGB")
        img_array = np.array(image, dtype=np.float32) / 255.0

        features = []
        self._feature_names = []

        if self.config.use_dct:
            dct_feats = self._extract_dct_features(img_array)
            features.append(dct_feats)
            self._feature_names.extend([f"dct_{i}" for i in range(len(dct_feats))])

        if self.config.use_noise:
            noise_feats = self._extract_noise_features(img_array)
            features.append(noise_feats)
            self._feature_names.extend([f"noise_{i}" for i in range(len(noise_feats))])

        if self.config.use_jpeg_artifacts:
            jpeg_feats = self._extract_jpeg_features(img_array)
            features.append(jpeg_feats)
            self._feature_names.extend([f"jpeg_{i}" for i in range(len(jpeg_feats))])

        if self.config.use_color_stats:
            color_feats = self._extract_color_features(img_array)
            features.append(color_feats)
            self._feature_names.extend([f"color_{i}" for i in range(len(color_feats))])

        if self.config.use_lbp:
            lbp_feats = self._extract_lbp_features(img_array)
            features.append(lbp_feats)
            self._feature_names.extend([f"lbp_{i}" for i in range(len(lbp_feats))])

        if self.config.use_edges:
            edge_feats = self._extract_edge_features(img_array)
            features.append(edge_feats)
            self._feature_names.extend([f"edge_{i}" for i in range(len(edge_feats))])

        return np.concatenate(features).astype(np.float32)

    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extract features from multiple images.

        Args:
            images: List of PIL Images

        Returns:
            2D numpy array of shape (n_images, n_features)
        """
        return np.stack([self.extract(img) for img in images])

    def _extract_dct_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extract DCT (Discrete Cosine Transform) features.

        GANs often leave fingerprints in the frequency domain.
        """
        try:
            from scipy.fftpack import dct
        except ImportError:
            # Fallback: simple FFT-based approximation
            return self._extract_fft_features(img)

        # Convert to grayscale
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        # Apply 2D DCT
        dct_2d = dct(dct(gray, axis=0, norm='ortho'), axis=1, norm='ortho')

        # Extract low-frequency coefficients (most informative)
        n = int(np.sqrt(self.config.dct_coeffs))
        coeffs = dct_2d[:n, :n].flatten()

        # Pad or truncate to exact size
        if len(coeffs) < self.config.dct_coeffs:
            coeffs = np.pad(coeffs, (0, self.config.dct_coeffs - len(coeffs)))
        else:
            coeffs = coeffs[:self.config.dct_coeffs]

        return coeffs

    def _extract_fft_features(self, img: np.ndarray) -> np.ndarray:
        """Fallback FFT features if scipy not available."""
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        # 2D FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # Radial average
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        max_radius = min(cy, cx)

        radial_profile = []
        for r in range(min(max_radius, self.config.dct_coeffs)):
            mask = self._create_ring_mask(h, w, cy, cx, r, r + 1)
            if mask.sum() > 0:
                radial_profile.append(magnitude[mask].mean())
            else:
                radial_profile.append(0)

        result = np.array(radial_profile[:self.config.dct_coeffs])
        if len(result) < self.config.dct_coeffs:
            result = np.pad(result, (0, self.config.dct_coeffs - len(result)))

        return result

    def _create_ring_mask(
        self, h: int, w: int, cy: int, cx: int, r_inner: int, r_outer: int
    ) -> np.ndarray:
        """Create a ring mask for radial averaging."""
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        return (dist >= r_inner) & (dist < r_outer)

    def _extract_noise_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extract noise residual features.

        Real camera images have characteristic sensor noise patterns.
        AI images often have different noise statistics.
        """
        # Estimate noise using high-pass filter
        from scipy.ndimage import convolve

        # Laplacian kernel for noise estimation
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

        features = []

        for c in range(3):  # RGB channels
            channel = img[:, :, c]

            try:
                residual = convolve(channel, kernel, mode='reflect')
            except Exception:
                # Fallback: simple difference
                residual = np.diff(channel, axis=0)[:-1, :] + np.diff(channel, axis=1)[:, :-1]

            # Noise statistics
            features.append(np.std(residual))
            features.append(np.mean(np.abs(residual)))

            # Histogram of noise magnitudes
            hist, _ = np.histogram(
                np.abs(residual).flatten(),
                bins=self.config.noise_sigma_bins // 3,
                range=(0, 0.1),
                density=True
            )
            features.extend(hist)

        # Cross-channel correlation (real cameras have correlated noise)
        features.append(self._noise_cross_correlation(img))

        return np.array(features, dtype=np.float32)

    def _noise_cross_correlation(self, img: np.ndarray) -> float:
        """Compute noise correlation across color channels."""
        try:
            from scipy.ndimage import convolve
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            residuals = [convolve(img[:, :, c], kernel, mode='reflect') for c in range(3)]
        except Exception:
            residuals = [np.diff(img[:, :, c], axis=0)[:-1, :] for c in range(3)]

        # Correlation between R-G and G-B residuals
        rg_corr = np.corrcoef(residuals[0].flatten(), residuals[1].flatten())[0, 1]
        gb_corr = np.corrcoef(residuals[1].flatten(), residuals[2].flatten())[0, 1]

        return (np.nan_to_num(rg_corr) + np.nan_to_num(gb_corr)) / 2

    def _extract_jpeg_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extract JPEG compression artifact features.

        JPEG uses 8x8 blocks; AI images may lack these artifacts.
        """
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        features = []

        # Block boundary discontinuity (JPEG blocking artifacts)
        h, w = gray.shape
        block_size = 8

        # Horizontal block boundaries
        h_boundaries = []
        for x in range(block_size, w - block_size, block_size):
            diff = np.abs(gray[:, x] - gray[:, x - 1])
            h_boundaries.append(np.mean(diff))

        # Vertical block boundaries
        v_boundaries = []
        for y in range(block_size, h - block_size, block_size):
            diff = np.abs(gray[y, :] - gray[y - 1, :])
            v_boundaries.append(np.mean(diff))

        features.append(np.mean(h_boundaries) if h_boundaries else 0)
        features.append(np.std(h_boundaries) if h_boundaries else 0)
        features.append(np.mean(v_boundaries) if v_boundaries else 0)
        features.append(np.std(v_boundaries) if v_boundaries else 0)

        # DCT coefficient histogram (first AC coefficient)
        # Simplified: measure local variance at block grid
        block_variances = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y + block_size, x:x + block_size]
                block_variances.append(np.var(block))

        features.append(np.mean(block_variances) if block_variances else 0)
        features.append(np.std(block_variances) if block_variances else 0)
        features.append(np.median(block_variances) if block_variances else 0)
        features.append(np.percentile(block_variances, 90) if block_variances else 0)

        return np.array(features, dtype=np.float32)

    def _extract_color_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extract color channel statistics.

        AI images may have different color distributions than natural photos.
        """
        features = []

        # Per-channel statistics
        for c in range(3):
            channel = img[:, :, c]
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.percentile(channel, 5),
                np.percentile(channel, 95),
            ])

        # Per-channel histograms
        for c in range(3):
            hist, _ = np.histogram(
                img[:, :, c].flatten(),
                bins=self.config.color_bins,
                range=(0, 1),
                density=True
            )
            features.extend(hist)

        return np.array(features, dtype=np.float32)

    def _extract_lbp_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extract Local Binary Pattern features.

        LBP captures texture patterns that differ between real and AI images.
        """
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        gray = (gray * 255).astype(np.uint8)

        try:
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(
                gray,
                P=self.config.lbp_points,
                R=self.config.lbp_radius,
                method='uniform'
            )
            n_bins = self.config.lbp_points + 2
        except ImportError:
            # Simplified LBP fallback
            lbp = self._simple_lbp(gray)
            n_bins = 2 ** self.config.lbp_points

        # LBP histogram
        hist, _ = np.histogram(
            lbp.flatten(),
            bins=n_bins,
            range=(0, n_bins),
            density=True
        )

        # Pad to expected size
        expected_size = 2 ** self.config.lbp_points
        if len(hist) < expected_size:
            hist = np.pad(hist, (0, expected_size - len(hist)))
        else:
            hist = hist[:expected_size]

        return hist.astype(np.float32)

    def _simple_lbp(self, gray: np.ndarray) -> np.ndarray:
        """Simplified LBP without skimage dependency."""
        h, w = gray.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

        # 8-neighbor offsets
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

        for i, (dy, dx) in enumerate(offsets):
            neighbor = gray[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
            center = gray[1:h - 1, 1:w - 1]
            lbp |= ((neighbor >= center).astype(np.uint8) << i)

        return lbp

    def _extract_edge_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extract edge statistics.

        Real images have natural edge distributions; AI may be too smooth or too sharp.
        """
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        # Sobel-like gradients
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)

        # Gradient magnitude
        min_shape = (min(gx.shape[0], gy.shape[0]), min(gx.shape[1], gy.shape[1]))
        gx = gx[:min_shape[0], :min_shape[1]]
        gy = gy[:min_shape[0], :min_shape[1]]

        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        direction = np.arctan2(gy, gx)

        features = [
            np.mean(magnitude),
            np.std(magnitude),
            np.percentile(magnitude, 90),
            np.percentile(magnitude, 99),
            np.mean(magnitude > 0.1),  # Edge density
        ]

        # Direction histogram (4 bins)
        dir_hist, _ = np.histogram(
            direction.flatten(),
            bins=4,
            range=(-np.pi, np.pi),
            density=True
        )
        features.extend(dir_hist[:3])  # Last bin is redundant

        return np.array(features, dtype=np.float32)
