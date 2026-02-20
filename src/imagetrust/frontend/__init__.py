"""
ImageTrust Frontend Module
==========================
Web (Streamlit) and Desktop (PySide6) interfaces.

Imports are lazy to avoid heavy dependency chains at package level
and to prevent PySide6/shiboken conflicts with pandas/six.
"""


def main():
    """Launch the Streamlit web frontend (lazy import)."""
    from imagetrust.frontend.app import main as _streamlit_main
    _streamlit_main()


def main_desktop():
    """Launch the PySide6 desktop frontend (lazy import)."""
    from imagetrust.frontend.pyside_app import main as _qt_main
    _qt_main()


__all__ = ["main", "main_desktop"]
