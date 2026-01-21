#!/usr/bin/env python3
"""
Build script for ImageTrust Desktop Application.

Creates a Windows executable using PyInstaller.

Usage:
    python scripts/build_desktop.py          # Folder distribution
    python scripts/build_desktop.py --onefile  # Single executable

Output:
    dist/ImageTrust/ImageTrust.exe  (folder mode)
    dist/ImageTrust.exe             (onefile mode)

Requirements:
    pip install 'imagetrust[desktop]'

Author: ImageTrust Team
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import PyInstaller
    except ImportError:
        missing.append("pyinstaller")

    try:
        import PySide6
    except ImportError:
        missing.append("PySide6")

    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print("  pip install 'imagetrust[desktop]'")
        print("  pip install pyinstaller")
        return False

    return True


def clean_build():
    """Clean previous build artifacts."""
    dirs_to_clean = ["build", "dist", "__pycache__"]

    for dir_name in dirs_to_clean:
        dir_path = PROJECT_ROOT / dir_name
        if dir_path.exists():
            print(f"Cleaning {dir_path}...")
            shutil.rmtree(dir_path)


def build(onefile: bool = False, clean: bool = True):
    """Build the desktop application."""
    print("=" * 60)
    print("ImageTrust Desktop Build")
    print("=" * 60)

    if not check_dependencies():
        return 1

    if clean:
        clean_build()

    spec_file = PROJECT_ROOT / "ImageTrust.spec"

    if not spec_file.exists():
        print(f"Error: Spec file not found: {spec_file}")
        return 1

    print(f"\nBuilding from: {spec_file}")
    print(f"Mode: {'Single file' if onefile else 'Folder distribution'}")
    print("-" * 60)

    # Build command
    cmd = [sys.executable, "-m", "PyInstaller", str(spec_file)]

    if onefile:
        cmd.append("--onefile")

    cmd.extend([
        "--noconfirm",  # Replace existing build
        "--clean",      # Clean cache
    ])

    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print("\nBuild failed!")
        return result.returncode

    # Check output
    if onefile:
        output = PROJECT_ROOT / "dist" / "ImageTrust.exe"
    else:
        output = PROJECT_ROOT / "dist" / "ImageTrust" / "ImageTrust.exe"

    if output.exists():
        size_mb = output.stat().st_size / (1024 * 1024)
        print("\n" + "=" * 60)
        print("BUILD SUCCESSFUL")
        print("=" * 60)
        print(f"Output: {output}")
        print(f"Size: {size_mb:.1f} MB")

        if not onefile:
            dist_folder = PROJECT_ROOT / "dist" / "ImageTrust"
            total_size = sum(f.stat().st_size for f in dist_folder.rglob("*") if f.is_file())
            print(f"Total distribution size: {total_size / (1024 * 1024):.1f} MB")

        print("\nTo run the application:")
        print(f"  {output}")
    else:
        print(f"\nWarning: Expected output not found: {output}")
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Build ImageTrust Desktop Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/build_desktop.py              # Folder distribution
    python scripts/build_desktop.py --onefile    # Single executable
    python scripts/build_desktop.py --no-clean   # Keep previous build artifacts
        """,
    )

    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Create single executable (slower startup)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Don't clean previous build artifacts",
    )

    args = parser.parse_args()

    return build(onefile=args.onefile, clean=not args.no_clean)


if __name__ == "__main__":
    sys.exit(main())
