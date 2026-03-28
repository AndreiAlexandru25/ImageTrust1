"""
Configuration management for ImageTrust.

Uses Pydantic-settings for robust and validated configuration,
allowing settings to be loaded from environment variables, .env files,
or default values.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings for ImageTrust.
    
    Settings can be configured via:
    - Environment variables (prefixed with IMAGETRUST_)
    - .env file
    - configs/default.yaml
    - Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="IMAGETRUST_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ===========================================
    # Project Settings
    # ===========================================
    project_name: str = "ImageTrust"
    project_version: str = "1.0.1"
    environment: Literal["development", "testing", "production"] = "development"

    # ===========================================
    # Logging
    # ===========================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: Optional[Path] = None

    # ===========================================
    # Paths
    # ===========================================
    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")
    outputs_dir: Path = Path("./outputs")
    reports_dir: Path = Path("./reports")
    temp_dir: Path = Path("./tmp")

    # ===========================================
    # Model Settings
    # ===========================================
    detector_backbone: str = "convnext_base"
    detector_pretrained: bool = True
    detector_checkpoint: Optional[Path] = None
    detector_device: Optional[str] = None  # "cuda", "cpu", "mps", or None for auto

    # Custom trained model (from Kaggle)
    kaggle_model_path: Optional[Path] = None  # Path to best_model.pth

    # Confidence bounds
    min_confidence: float = Field(default=0.80, ge=0.5, le=1.0)
    max_confidence: float = Field(default=0.95, ge=0.5, le=1.0)

    # Ensemble settings
    ensemble_enabled: bool = True
    ensemble_strategy: Literal["average", "weighted", "voting", "stacking"] = "weighted"
    ensemble_models: List[str] = ["efficientnet_b0", "convnext_base", "vit_base_patch16_224"]
    ensemble_weights: Optional[List[float]] = None

    # ===========================================
    # Preprocessing
    # ===========================================
    input_size: int = 224
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ===========================================
    # API Settings
    # ===========================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    max_upload_size_mb: int = 50
    cors_origins: List[str] = ["*"]

    # ===========================================
    # Frontend Settings
    # ===========================================
    frontend_port: int = 8501

    # ===========================================
    # Explainability Settings
    # ===========================================
    gradcam_enabled: bool = True
    gradcam_target_layer: Optional[str] = None
    patch_analysis_enabled: bool = True
    patch_size: int = 64
    patch_stride: int = 32

    # ===========================================
    # Evaluation Settings
    # ===========================================
    eval_batch_size: int = 8
    eval_num_workers: int = 4

    # ===========================================
    # Validators
    # ===========================================
    @field_validator("min_confidence", "max_confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.5 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.5 and 1.0")
        return v

    # ===========================================
    # Methods
    # ===========================================
    def get_device(self) -> torch.device:
        """Determine the appropriate torch device."""
        if self.detector_device:
            return torch.device(self.detector_device)
        
        if torch.cuda.is_available():
            return torch.device("cuda")
        
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        
        return torch.device("cpu")

    def ensure_dirs(self) -> None:
        """Ensure all necessary directories exist."""
        for dir_path in [
            self.data_dir,
            self.models_dir,
            self.outputs_dir,
            self.reports_dir,
            self.temp_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str) -> Path:
        """Get path to a model checkpoint."""
        return self.models_dir / f"{model_name}.pth"


@lru_cache()
def get_settings() -> Settings:
    """
    Get the cached settings instance.
    
    Uses LRU cache to ensure settings are only loaded once.
    """
    settings = Settings()
    settings.ensure_dirs()
    return settings
