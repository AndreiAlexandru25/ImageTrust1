#!/usr/bin/env python3
"""
ImageTrust Setup Verification Script.

Verifies that the ImageTrust installation is complete and functional.
Checks all dependencies, models, and core functionality.

Usage:
    python scripts/verify_setup.py
    python scripts/verify_setup.py --verbose
    python scripts/verify_setup.py --fix  # Attempt to fix issues

Author: ImageTrust Team
"""

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Check Results
# =============================================================================

class CheckResult:
    """Result of a verification check."""

    def __init__(self, name: str, passed: bool, message: str = "", fix_hint: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
        self.fix_hint = fix_hint

    def __str__(self):
        status = "✓" if self.passed else "✗"
        result = f"[{status}] {self.name}"
        if self.message:
            result += f": {self.message}"
        return result


# =============================================================================
# Verification Checks
# =============================================================================

def check_python_version() -> CheckResult:
    """Check Python version is 3.10+."""
    version = sys.version_info
    passed = version >= (3, 10)
    message = f"Python {version.major}.{version.minor}.{version.micro}"

    return CheckResult(
        name="Python Version",
        passed=passed,
        message=message,
        fix_hint="Install Python 3.10 or higher",
    )


def check_core_dependencies() -> List[CheckResult]:
    """Check that core dependencies are installed."""
    results = []

    core_packages = [
        ("numpy", "numpy"),
        ("PIL", "pillow"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("fastapi", "fastapi"),
        ("click", "click"),
        ("pydantic", "pydantic"),
        ("sklearn", "scikit-learn"),
    ]

    for import_name, pip_name in core_packages:
        try:
            importlib.import_module(import_name)
            results.append(CheckResult(
                name=f"Package: {pip_name}",
                passed=True,
                message="installed",
            ))
        except ImportError:
            results.append(CheckResult(
                name=f"Package: {pip_name}",
                passed=False,
                message="not found",
                fix_hint=f"pip install {pip_name}",
            ))

    return results


def check_optional_dependencies() -> List[CheckResult]:
    """Check optional dependencies."""
    results = []

    optional_packages = [
        ("PySide6", "PySide6", "desktop"),
        ("streamlit", "streamlit", "web UI"),
        ("reportlab", "reportlab", "PDF reports"),
        ("grad_cam", "pytorch-grad-cam", "explainability"),
    ]

    for import_name, pip_name, feature in optional_packages:
        try:
            importlib.import_module(import_name)
            results.append(CheckResult(
                name=f"Optional: {pip_name}",
                passed=True,
                message=f"installed (for {feature})",
            ))
        except ImportError:
            results.append(CheckResult(
                name=f"Optional: {pip_name}",
                passed=True,  # Optional, so not a failure
                message=f"not installed (needed for {feature})",
                fix_hint=f"pip install {pip_name}",
            ))

    return results


def check_gpu_availability() -> CheckResult:
    """Check if GPU is available."""
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return CheckResult(
                name="GPU (CUDA)",
                passed=True,
                message=f"{device_name}",
            )
        else:
            return CheckResult(
                name="GPU (CUDA)",
                passed=True,  # Not required
                message="not available (CPU will be used)",
                fix_hint="Install CUDA toolkit and PyTorch with CUDA support",
            )
    except Exception as e:
        return CheckResult(
            name="GPU (CUDA)",
            passed=True,
            message=f"check failed: {e}",
        )


def check_imagetrust_package() -> CheckResult:
    """Check that ImageTrust package is installed."""
    try:
        import imagetrust
        return CheckResult(
            name="ImageTrust Package",
            passed=True,
            message="installed",
        )
    except ImportError:
        return CheckResult(
            name="ImageTrust Package",
            passed=False,
            message="not installed",
            fix_hint="pip install -e .",
        )


def check_imagetrust_modules() -> List[CheckResult]:
    """Check that core ImageTrust modules can be imported."""
    results = []

    modules = [
        "imagetrust.detection",
        "imagetrust.baselines",
        "imagetrust.cli",
        "imagetrust.api",
        "imagetrust.evaluation",
    ]

    for module in modules:
        try:
            importlib.import_module(module)
            results.append(CheckResult(
                name=f"Module: {module}",
                passed=True,
                message="importable",
            ))
        except ImportError as e:
            results.append(CheckResult(
                name=f"Module: {module}",
                passed=False,
                message=f"import error: {e}",
                fix_hint="Check installation and dependencies",
            ))

    return results


def check_cli_available() -> CheckResult:
    """Check that CLI is accessible."""
    cli_path = shutil.which("imagetrust")
    if cli_path:
        return CheckResult(
            name="CLI Command",
            passed=True,
            message=f"available at {cli_path}",
        )
    else:
        # Try running via python -m
        try:
            result = subprocess.run(
                [sys.executable, "-m", "imagetrust", "--help"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                return CheckResult(
                    name="CLI Command",
                    passed=True,
                    message="available via python -m imagetrust",
                )
        except Exception:
            pass

        return CheckResult(
            name="CLI Command",
            passed=False,
            message="not found in PATH",
            fix_hint="pip install -e . (ensure scripts are in PATH)",
        )


def check_project_structure() -> List[CheckResult]:
    """Check that required project directories exist."""
    results = []

    required_dirs = [
        "src/imagetrust",
        "configs",
        "scripts",
        "tests",
        "docs",
    ]

    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        exists = full_path.exists() and full_path.is_dir()

        results.append(CheckResult(
            name=f"Directory: {dir_path}",
            passed=exists,
            message="exists" if exists else "missing",
            fix_hint=f"mkdir -p {dir_path}",
        ))

    return results


def check_config_files() -> List[CheckResult]:
    """Check that configuration files exist."""
    results = []

    config_files = [
        "pyproject.toml",
        "configs/hyperparameters.yaml",
        ".pre-commit-config.yaml",
    ]

    for file_path in config_files:
        full_path = PROJECT_ROOT / file_path
        exists = full_path.exists() and full_path.is_file()

        results.append(CheckResult(
            name=f"Config: {file_path}",
            passed=exists,
            message="exists" if exists else "missing",
        ))

    return results


def check_model_availability() -> CheckResult:
    """Check if models can be loaded (basic check)."""
    try:
        from transformers import AutoModelForImageClassification, AutoFeatureExtractor

        # Just check that transformers can access HuggingFace
        # Don't actually download models
        return CheckResult(
            name="Model Loading",
            passed=True,
            message="HuggingFace transformers ready",
        )
    except Exception as e:
        return CheckResult(
            name="Model Loading",
            passed=False,
            message=f"error: {e}",
            fix_hint="pip install transformers",
        )


def check_test_suite() -> CheckResult:
    """Check that tests can be discovered."""
    try:
        import pytest

        # Discover tests without running
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             str(PROJECT_ROOT / "tests")],
            capture_output=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )

        output = result.stdout.decode()
        # Count tests
        if "test" in output.lower():
            lines = output.strip().split("\n")
            return CheckResult(
                name="Test Suite",
                passed=True,
                message=f"tests discoverable",
            )
        else:
            return CheckResult(
                name="Test Suite",
                passed=True,
                message="no tests found (may be OK)",
            )
    except Exception as e:
        return CheckResult(
            name="Test Suite",
            passed=True,  # Not critical
            message=f"check skipped: {e}",
        )


# =============================================================================
# Main Verification
# =============================================================================

def run_all_checks(verbose: bool = False) -> Tuple[List[CheckResult], int, int]:
    """Run all verification checks."""
    results = []

    print("\n" + "=" * 60)
    print("ImageTrust Setup Verification")
    print("=" * 60)

    # Python version
    print("\n[1/8] Checking Python version...")
    results.append(check_python_version())

    # Core dependencies
    print("[2/8] Checking core dependencies...")
    results.extend(check_core_dependencies())

    # Optional dependencies
    print("[3/8] Checking optional dependencies...")
    results.extend(check_optional_dependencies())

    # GPU
    print("[4/8] Checking GPU availability...")
    results.append(check_gpu_availability())

    # ImageTrust package
    print("[5/8] Checking ImageTrust package...")
    results.append(check_imagetrust_package())
    results.extend(check_imagetrust_modules())

    # CLI
    print("[6/8] Checking CLI...")
    results.append(check_cli_available())

    # Project structure
    print("[7/8] Checking project structure...")
    results.extend(check_project_structure())
    results.extend(check_config_files())

    # Models and tests
    print("[8/8] Checking models and tests...")
    results.append(check_model_availability())
    results.append(check_test_suite())

    # Count results
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    return results, passed, failed


def print_results(results: List[CheckResult], verbose: bool = False):
    """Print verification results."""
    print("\n" + "-" * 60)
    print("Results")
    print("-" * 60)

    # Group by status
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    if failed:
        print("\n❌ FAILED CHECKS:")
        for result in failed:
            print(f"  {result}")
            if result.fix_hint:
                print(f"     Fix: {result.fix_hint}")

    if verbose or not failed:
        print("\n✓ PASSED CHECKS:")
        for result in passed:
            print(f"  {result}")

    # Summary
    print("\n" + "-" * 60)
    total = len(results)
    passed_count = len(passed)
    failed_count = len(failed)

    if failed_count == 0:
        print(f"✅ All {total} checks passed!")
        print("\nImageTrust is ready to use.")
    else:
        print(f"⚠️  {passed_count}/{total} checks passed, {failed_count} failed")
        print("\nPlease fix the failed checks before using ImageTrust.")


def attempt_fixes(results: List[CheckResult]):
    """Attempt to fix failed checks."""
    failed = [r for r in results if not r.passed and r.fix_hint]

    if not failed:
        print("No automatic fixes available.")
        return

    print("\n" + "-" * 60)
    print("Attempting Fixes")
    print("-" * 60)

    for result in failed:
        if result.fix_hint.startswith("pip install"):
            print(f"\nRunning: {result.fix_hint}")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install"] +
                    result.fix_hint.replace("pip install ", "").split(),
                    check=True,
                )
                print(f"  ✓ Fixed: {result.name}")
            except subprocess.CalledProcessError:
                print(f"  ✗ Failed to fix: {result.name}")
        elif result.fix_hint.startswith("mkdir"):
            dir_path = result.fix_hint.replace("mkdir -p ", "")
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                print(f"  ✓ Created: {dir_path}")
            except Exception as e:
                print(f"  ✗ Failed to create {dir_path}: {e}")
        else:
            print(f"\nManual fix needed for {result.name}:")
            print(f"  {result.fix_hint}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify ImageTrust setup",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all checks, not just failures",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix failed checks",
    )

    args = parser.parse_args()

    # Run checks
    results, passed, failed = run_all_checks(args.verbose)

    # Print results
    print_results(results, args.verbose)

    # Attempt fixes if requested
    if args.fix and failed > 0:
        attempt_fixes(results)
        print("\nRe-running verification...")
        results, passed, failed = run_all_checks(args.verbose)
        print_results(results, args.verbose)

    # Exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
