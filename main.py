#!/usr/bin/env python3
"""
ImageTrust - Main Entry Point.

A Forensic Application for Identifying AI-Generated
and Digitally Manipulated Images.

Usage:
    # Analyze a single image
    python main.py analyze image.jpg
    
    # Start the API server
    python main.py serve
    
    # Launch the web UI
    python main.py ui
    
    # Run evaluation
    python main.py evaluate --dataset ./testset/

For CLI help:
    python main.py --help
    python main.py analyze --help
"""

import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from imagetrust.cli import main

if __name__ == "__main__":
    main()
