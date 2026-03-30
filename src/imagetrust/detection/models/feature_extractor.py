"""
Feature Extraction Module.

Provides multi-level feature extraction for AI image detection including:
- Visual features (CNN/ViT embeddings)
- Frequency domain features (FFT patterns)
- Noise patterns (sensor noise analysis)
- Statistical features (color, texture)
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


class FrequencyFeatureExtractor(nn.Module):
    """
    Frequency domain feature extraction.
    
    AI-generated images often have distinctive patterns in the
    frequency domain that differ from real photographs.
    
    Features extracted:
    - FFT magnitude spectrum
    - Azimuthal average of power spectrum
    - High/low frequency ratio
    - Spectral centroid
    """
    
    def __init__(
        self,
        output_dim: int = 128,
        use_phase: bool = True,
    ) -> None:
        """
        Initialize frequency feature extractor.
        
        Args:
            output_dim: Output feature dimension
            use_phase: Include phase information
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.use_phase = use_phase
        
        # Input channels: 3 (magnitude) + 3 (phase) if use_phase
        in_channels = 6 if use_phase else 3
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
        )
    
    def compute_fft(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute FFT magnitude and phase."""
        # Apply 2D FFT
        fft = torch.fft.fft2(x, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft)
        
        # Magnitude (log scale for better visualization)
        magnitude = torch.abs(fft_shifted)
        magnitude = torch.log1p(magnitude)
        
        # Phase
        phase = torch.angle(fft_shifted)
        
        return magnitude, phase
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency features.
        
        Args:
            x: Input images (N, C, H, W)
        
        Returns:
            Frequency features (N, output_dim)
        """
        magnitude, phase = self.compute_fft(x)
        
        if self.use_phase:
            freq_input = torch.cat([magnitude, phase], dim=1)
        else:
            freq_input = magnitude
        
        encoded = self.encoder(freq_input)
        features = self.fc(encoded)
        
        return features
    
    def extract_spectral_features(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Extract interpretable spectral features.
        
        Args:
            x: Input images
        
        Returns:
            Dictionary with spectral features
        """
        magnitude, phase = self.compute_fft(x)
        
        batch_size = x.size(0)
        features = {}
        
        # Convert to numpy for easier computation
        mag_np = magnitude.detach().cpu().numpy()
        
        # Compute azimuthal average (radial power spectrum)
        h, w = mag_np.shape[2:]
        center_y, center_x = h // 2, w // 2
        
        y, x_grid = np.ogrid[:h, :w]
        r = np.sqrt((x_grid - center_x)**2 + (y - center_y)**2)
        r = r.astype(int)
        
        max_r = min(center_x, center_y)
        
        radial_profiles = []
        for b in range(batch_size):
            # Average across color channels
            mag_avg = mag_np[b].mean(axis=0)
            
            # Compute radial profile
            radial_prof = np.zeros(max_r)
            for radius in range(max_r):
                mask = (r == radius)
                if mask.sum() > 0:
                    radial_prof[radius] = mag_avg[mask].mean()
            
            radial_profiles.append(radial_prof)
        
        features["radial_profile"] = torch.tensor(
            np.array(radial_profiles),
            dtype=torch.float32,
            device=magnitude.device,
        )
        
        # High/low frequency ratio
        low_freq_mask = r < max_r // 4
        high_freq_mask = r > max_r // 2
        
        low_freq_energy = []
        high_freq_energy = []
        
        for b in range(batch_size):
            mag_avg = mag_np[b].mean(axis=0)
            low_freq_energy.append(mag_avg[low_freq_mask].sum())
            high_freq_energy.append(mag_avg[high_freq_mask].sum())
        
        low_freq_energy = np.array(low_freq_energy)
        high_freq_energy = np.array(high_freq_energy)
        
        features["hf_lf_ratio"] = torch.tensor(
            high_freq_energy / (low_freq_energy + 1e-8),
            dtype=torch.float32,
            device=magnitude.device,
        )
        
        return features


class NoiseFeatureExtractor(nn.Module):
    """
    Noise pattern feature extraction.
    
    Real photographs contain characteristic sensor noise patterns
    that AI-generated images typically lack or have different patterns.
    
    Methods:
    - High-pass filtering for noise residual
    - Local variance estimation
    - PRNU (Photo Response Non-Uniformity) approximation
    """
    
    def __init__(
        self,
        output_dim: int = 128,
    ) -> None:
        """
        Initialize noise feature extractor.
        
        Args:
            output_dim: Output feature dimension
        """
        super().__init__()
        
        self.output_dim = output_dim
        
        # High-pass kernels for noise extraction
        self.register_buffer("laplacian_kernel", torch.tensor([
            [[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]],
            [[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]],
            [[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]],
        ], dtype=torch.float32))
        
        # Sobel kernels for edge detection
        self.register_buffer("sobel_x", torch.tensor([
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
        ], dtype=torch.float32))
        
        self.register_buffer("sobel_y", torch.tensor([
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
        ], dtype=torch.float32))
        
        # Noise encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
        )
    
    def extract_noise_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Extract noise residual using high-pass filtering."""
        batch_size = x.size(0)
        
        noise_residuals = []
        for i in range(3):  # For each color channel
            channel = x[:, i:i+1, :, :]
            residual = F.conv2d(channel, self.laplacian_kernel[:1], padding=1)
            noise_residuals.append(residual)
        
        noise_residual = torch.cat(noise_residuals, dim=1)
        return noise_residual
    
    def extract_edge_response(self, x: torch.Tensor) -> torch.Tensor:
        """Extract edge responses using Sobel filters."""
        edge_responses = []
        
        for i in range(3):
            channel = x[:, i:i+1, :, :]
            edge_x = F.conv2d(channel, self.sobel_x[:1], padding=1)
            edge_y = F.conv2d(channel, self.sobel_y[:1], padding=1)
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
            edge_responses.append(edge_magnitude)
        
        edges = torch.cat(edge_responses, dim=1)
        return edges
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract noise features.
        
        Args:
            x: Input images (N, C, H, W)
        
        Returns:
            Noise features (N, output_dim)
        """
        noise_residual = self.extract_noise_residual(x)
        edge_response = self.extract_edge_response(x)
        
        combined = torch.cat([noise_residual, edge_response], dim=1)
        
        encoded = self.encoder(combined)
        features = self.fc(encoded)
        
        return features
    
    def compute_local_variance(
        self,
        x: torch.Tensor,
        window_size: int = 5,
    ) -> torch.Tensor:
        """
        Compute local variance map.
        
        Args:
            x: Input images
            window_size: Window size for variance computation
        
        Returns:
            Local variance map
        """
        padding = window_size // 2
        
        # Compute local mean
        kernel = torch.ones(1, 1, window_size, window_size, device=x.device)
        kernel = kernel / (window_size ** 2)
        
        local_means = []
        local_vars = []
        
        for i in range(x.size(1)):
            channel = x[:, i:i+1, :, :]
            local_mean = F.conv2d(channel, kernel, padding=padding)
            local_sq_mean = F.conv2d(channel ** 2, kernel, padding=padding)
            local_var = local_sq_mean - local_mean ** 2
            local_vars.append(local_var)
        
        return torch.cat(local_vars, dim=1)


class StatisticalFeatureExtractor:
    """
    Statistical feature extraction (non-learnable).
    
    Extracts hand-crafted statistical features that can help
    distinguish AI-generated from real images.
    """
    
    @staticmethod
    def extract(image: Image.Image | np.ndarray) -> dict[str, float]:
        """
        Extract statistical features from image.
        
        Args:
            image: Input image
        
        Returns:
            Dictionary of statistical features
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert("RGB"))
        else:
            img_array = image
        
        features = {}
        
        # Normalize to [0, 1]
        img_float = img_array.astype(np.float32) / 255.0
        
        # Color statistics per channel
        for i, channel_name in enumerate(["red", "green", "blue"]):
            channel = img_float[:, :, i]
            features[f"{channel_name}_mean"] = float(np.mean(channel))
            features[f"{channel_name}_std"] = float(np.std(channel))
            features[f"{channel_name}_skewness"] = float(
                np.mean((channel - np.mean(channel))**3) / (np.std(channel)**3 + 1e-8)
            )
            features[f"{channel_name}_kurtosis"] = float(
                np.mean((channel - np.mean(channel))**4) / (np.std(channel)**4 + 1e-8)
            )
        
        # Color histogram entropy
        for i, channel_name in enumerate(["red", "green", "blue"]):
            hist, _ = np.histogram(img_array[:, :, i], bins=256, range=(0, 256))
            hist = hist / (hist.sum() + 1e-8)
            entropy = -np.sum(hist * np.log2(hist + 1e-8))
            features[f"{channel_name}_entropy"] = float(entropy)
        
        # Edge density (Canny-like)
        gray = np.mean(img_float, axis=2)
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        edge_magnitude = np.sqrt(gx[:, :-1]**2 + gy[:-1, :]**2)
        features["edge_density"] = float(np.mean(edge_magnitude > 0.1))
        features["edge_mean"] = float(np.mean(edge_magnitude))
        
        # Saturation statistics
        max_rgb = np.max(img_float, axis=2)
        min_rgb = np.min(img_float, axis=2)
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-8)
        features["saturation_mean"] = float(np.mean(saturation))
        features["saturation_std"] = float(np.std(saturation))
        
        # JPEG artifact detection (blocking)
        if img_array.shape[0] >= 8 and img_array.shape[1] >= 8:
            # Check for 8x8 block boundaries
            h, w = img_array.shape[:2]
            block_diffs = []
            
            for y in range(7, h - 1, 8):
                for x in range(w):
                    diff = abs(int(img_array[y, x, 0]) - int(img_array[y + 1, x, 0]))
                    block_diffs.append(diff)
            
            features["blocking_artifact"] = float(np.mean(block_diffs)) if block_diffs else 0.0
        else:
            features["blocking_artifact"] = 0.0
        
        return features


class MultiModalFeatureExtractor(nn.Module):
    """
    Combined multi-modal feature extractor.
    
    Combines:
    - Visual features from CNN/ViT backbone
    - Frequency domain features
    - Noise pattern features
    - Optional: statistical features
    """
    
    def __init__(
        self,
        visual_dim: int = 768,
        frequency_dim: int = 128,
        noise_dim: int = 128,
        output_dim: int = 512,
    ) -> None:
        """
        Initialize multi-modal feature extractor.
        
        Args:
            visual_dim: Visual feature dimension (from backbone)
            frequency_dim: Frequency feature dimension
            noise_dim: Noise feature dimension
            output_dim: Final output dimension
        """
        super().__init__()
        
        self.frequency_extractor = FrequencyFeatureExtractor(output_dim=frequency_dim)
        self.noise_extractor = NoiseFeatureExtractor(output_dim=noise_dim)
        
        total_dim = visual_dim + frequency_dim + noise_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        self.output_dim = output_dim
    
    def forward(
        self,
        x: torch.Tensor,
        visual_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Extract multi-modal features.
        
        Args:
            x: Input images (N, C, H, W)
            visual_features: Pre-computed visual features (N, visual_dim)
        
        Returns:
            Fused features (N, output_dim)
        """
        # Extract frequency features
        freq_features = self.frequency_extractor(x)
        
        # Extract noise features
        noise_features = self.noise_extractor(x)
        
        # Combine all features
        if visual_features is not None:
            all_features = torch.cat([
                visual_features,
                freq_features,
                noise_features,
            ], dim=1)
        else:
            all_features = torch.cat([freq_features, noise_features], dim=1)
        
        # Fuse features
        fused = self.fusion(all_features)
        
        return fused
    
    def get_feature_breakdown(
        self,
        x: torch.Tensor,
        visual_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Get individual feature components.
        
        Args:
            x: Input images
            visual_features: Pre-computed visual features
        
        Returns:
            Dictionary with feature components
        """
        freq_features = self.frequency_extractor(x)
        noise_features = self.noise_extractor(x)
        
        return {
            "visual": visual_features,
            "frequency": freq_features,
            "noise": noise_features,
            "fused": self.forward(x, visual_features),
        }
