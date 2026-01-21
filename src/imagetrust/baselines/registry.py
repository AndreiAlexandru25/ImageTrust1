"""
Baseline registry and factory.

Provides unified access to all baseline implementations.
"""

from typing import Any, Dict, List, Optional, Type

from imagetrust.baselines.base import BaselineDetector, BaselineConfig


# Global registry
BASELINE_REGISTRY: Dict[str, Type[BaselineDetector]] = {}


def register_baseline(name: str):
    """
    Decorator to register a baseline class.

    Example:
        @register_baseline("classical")
        class ClassicalBaseline(BaselineDetector):
            ...
    """
    def decorator(cls: Type[BaselineDetector]):
        BASELINE_REGISTRY[name] = cls
        return cls
    return decorator


def list_baselines() -> List[str]:
    """List all registered baseline names."""
    _register_all_baselines()
    return list(BASELINE_REGISTRY.keys())


def get_baseline(
    name: str,
    config: Optional[BaselineConfig] = None,
    **kwargs,
) -> BaselineDetector:
    """
    Create a baseline instance by name.

    Args:
        name: Baseline name (classical, cnn, vit, imagetrust, or alias)
        config: Optional configuration (created from defaults if not provided)
        **kwargs: Additional arguments passed to baseline constructor

    Returns:
        BaselineDetector instance

    Example:
        >>> baseline = get_baseline("classical", classifier="xgboost")
        >>> baseline = get_baseline("cnn", backbone="efficientnet_b0")
        >>> baseline = get_baseline("vit", architecture="clip")
        >>> baseline = get_baseline("imagetrust")  # Our method
    """
    # Resolve aliases
    name_lower = name.lower()
    aliases = {
        "b1": "classical",
        "logreg": "classical",
        "xgboost": "classical",
        "feature": "classical",
        "b2": "cnn",
        "resnet": "cnn",
        "resnet50": "cnn",
        "efficientnet": "cnn",
        "b3": "vit",
        "transformer": "vit",
        "clip": "vit",
        "ours": "imagetrust",
        "ensemble": "imagetrust",
        "multi": "imagetrust",
    }

    resolved_name = aliases.get(name_lower, name_lower)

    if resolved_name not in BASELINE_REGISTRY:
        # Lazy import and register if not already done
        _register_all_baselines()

    if resolved_name not in BASELINE_REGISTRY:
        available = list(BASELINE_REGISTRY.keys())
        raise ValueError(
            f"Unknown baseline: {name}. Available: {available}"
        )

    # Create default config if not provided
    if config is None:
        config = _get_default_config(resolved_name)

    # Get baseline class
    baseline_cls = BASELINE_REGISTRY[resolved_name]

    return baseline_cls(config, **kwargs)


def _register_all_baselines() -> None:
    """Register all baseline classes."""
    # Import to trigger registration
    from imagetrust.baselines.classical_baseline import ClassicalBaseline
    from imagetrust.baselines.cnn_baseline import CNNBaseline
    from imagetrust.baselines.vit_baseline import ViTBaseline
    from imagetrust.baselines.imagetrust_wrapper import ImageTrustWrapper

    # Manual registration (in case decorators not used)
    BASELINE_REGISTRY["classical"] = ClassicalBaseline
    BASELINE_REGISTRY["cnn"] = CNNBaseline
    BASELINE_REGISTRY["vit"] = ViTBaseline
    BASELINE_REGISTRY["imagetrust"] = ImageTrustWrapper


def _get_default_config(baseline_name: str) -> BaselineConfig:
    """Get default configuration for a baseline."""
    defaults = {
        "classical": BaselineConfig(
            name="Classical (LogReg)",
            seed=42,
            epochs=1,  # Not used for classical
            batch_size=1,
        ),
        "cnn": BaselineConfig(
            name="CNN (ResNet-50)",
            seed=42,
            epochs=10,
            batch_size=32,
            learning_rate=1e-4,
            weight_decay=1e-4,
        ),
        "vit": BaselineConfig(
            name="ViT-B/16",
            seed=42,
            epochs=10,
            batch_size=16,
            learning_rate=1e-5,  # Lower for ViT
            weight_decay=1e-2,
        ),
        "imagetrust": BaselineConfig(
            name="ImageTrust (Ours)",
            seed=42,
            epochs=0,  # No training - uses pretrained HF models
            batch_size=1,
        ),
    }

    return defaults.get(baseline_name, BaselineConfig(name=baseline_name))


def get_baseline_from_yaml(yaml_path: str, baseline_name: str) -> BaselineDetector:
    """
    Create a baseline from YAML configuration.

    Args:
        yaml_path: Path to baselines.yaml
        baseline_name: Which baseline to load (classical, cnn, vit)

    Returns:
        Configured baseline instance
    """
    import yaml
    from pathlib import Path

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Extract reproducibility settings
    repro = cfg.get("reproducibility", {})
    seed = repro.get("seed", 42)

    # Get baseline-specific config
    baseline_cfg = cfg.get(baseline_name, {})

    if baseline_name == "classical":
        config = BaselineConfig(
            name=baseline_cfg.get("name", "Classical (LogReg)"),
            seed=seed,
        )

        classifier = baseline_cfg.get("classifier", "logistic_regression")
        classifier_kwargs = baseline_cfg.get(classifier, {})

        # Build feature config
        from imagetrust.baselines.feature_extraction import FeatureConfig
        feat_cfg = baseline_cfg.get("features", {})
        feature_config = FeatureConfig(
            use_dct=feat_cfg.get("use_dct", True),
            dct_coeffs=feat_cfg.get("dct_coeffs", 64),
            use_noise=feat_cfg.get("use_noise", True),
            noise_sigma_bins=feat_cfg.get("noise_sigma_bins", 10),
            use_jpeg_artifacts=feat_cfg.get("use_jpeg_artifacts", True),
            use_color_stats=feat_cfg.get("use_color_stats", True),
            color_bins=feat_cfg.get("color_bins", 32),
            use_lbp=feat_cfg.get("use_lbp", True),
            lbp_radius=feat_cfg.get("lbp_radius", 1),
            lbp_points=feat_cfg.get("lbp_points", 8),
            use_edges=feat_cfg.get("use_edges", True),
        )

        return get_baseline(
            "classical",
            config=config,
            classifier=classifier,
            feature_config=feature_config,
            **classifier_kwargs,
        )

    elif baseline_name == "cnn":
        config = BaselineConfig(
            name=baseline_cfg.get("name", "CNN (ResNet-50)"),
            seed=seed,
            epochs=baseline_cfg.get("epochs", 10),
            batch_size=baseline_cfg.get("batch_size", 32),
            learning_rate=baseline_cfg.get("learning_rate", 1e-4),
            weight_decay=baseline_cfg.get("weight_decay", 1e-4),
        )

        return get_baseline(
            "cnn",
            config=config,
            backbone=baseline_cfg.get("backbone", "resnet50"),
            pretrained=baseline_cfg.get("pretrained", True),
            freeze_backbone=baseline_cfg.get("freeze_backbone", False),
            input_size=baseline_cfg.get("input_size", 224),
        )

    elif baseline_name == "vit":
        arch = baseline_cfg.get("architecture", "vit")
        arch_cfg = baseline_cfg.get(arch, {})

        config = BaselineConfig(
            name=baseline_cfg.get("name", "ViT-B/16"),
            seed=seed,
            epochs=baseline_cfg.get("epochs", 10),
            batch_size=baseline_cfg.get("batch_size", 16),
            learning_rate=baseline_cfg.get("learning_rate", 1e-5),
            weight_decay=baseline_cfg.get("weight_decay", 1e-2),
        )

        return get_baseline(
            "vit",
            config=config,
            architecture=arch,
            model_name=arch_cfg.get("model_name"),
            freeze_backbone=arch_cfg.get("freeze_backbone", False),
            input_size=baseline_cfg.get("input_size", 224),
        )

    else:
        raise ValueError(f"Unknown baseline in YAML: {baseline_name}")
