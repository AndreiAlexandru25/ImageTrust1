# -*- mode: python ; coding: utf-8 -*-
"""
ImageTrust Desktop (PySide6) - PyInstaller Spec File

Build command:
    pyinstaller --noconfirm ImageTrust.spec

Output:
    dist/ImageTrust/ImageTrust.exe
"""

import os

block_cipher = None

ROOT = os.path.abspath(".")
ENTRY_POINT = os.path.join(ROOT, "src", "imagetrust", "frontend", "pyside_app.py")

datas = [
    (os.path.join(ROOT, "configs"), "configs"),
    (os.path.join(ROOT, "assets"), "assets"),
]
datas = [(src, dst) for src, dst in datas if os.path.exists(src)]

hiddenimports = [
    "imagetrust",
    "imagetrust.frontend",
    "imagetrust.frontend.pyside_app",
    "imagetrust.detection",
    "imagetrust.detection.multi_detector",
    "imagetrust.detection.calibration",
    "imagetrust.detection.preprocessing",
    "imagetrust.detection.models",
    "imagetrust.detection.models.calibrated_ensemble",
    "imagetrust.detection.models.kaggle_detector",
    "imagetrust.utils",
    "imagetrust.utils.scoring",
    "imagetrust.core",
    "imagetrust.core.config",
    "imagetrust.core.types",
    "imagetrust.metadata",
    "imagetrust.baselines",
    "imagetrust.baselines.uncertainty",
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PIL",
    "PIL.Image",
    "numpy",
    "scipy",
    "scipy.ndimage",
    "sklearn",
    "sklearn.calibration",
    "torch",
    "torchvision",
    "transformers",
    "timm",
    "exifread",
    "pydantic",
    "pydantic_settings",
    "yaml",
    "loguru",
    "tqdm",
    "cv2",
]

excludes = [
    "streamlit",
    "matplotlib",
    "pandas",
    "reportlab",
    "jinja2",
    "uvicorn",
    "fastapi",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "black",
    "isort",
    "mypy",
    "flake8",
]

a = Analysis(
    [ENTRY_POINT],
    pathex=[os.path.join(ROOT, "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

icon_path = os.path.join(ROOT, "assets", "icon.ico")

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ImageTrust",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path if os.path.exists(icon_path) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ImageTrust",
)
