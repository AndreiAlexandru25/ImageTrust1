"""
ImageTrust Desktop Application (PySide6/Qt).

Modern cross-platform desktop GUI for AI image forensics.
Designed for offline use and Windows .exe packaging with PyInstaller.
"""

from imagetrust.desktop.app import ImageTrustApp, main

__all__ = ["ImageTrustApp", "main"]
