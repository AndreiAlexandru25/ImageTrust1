# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for ImageTrust Desktop Application.

Usage:
    # Build folder distribution (recommended)
    pyinstaller ImageTrust.spec

    # Build single-file executable (slower startup, larger file)
    pyinstaller ImageTrust.spec --onefile

Output:
    dist/ImageTrust/ImageTrust.exe  (folder mode)
    dist/ImageTrust.exe             (onefile mode)

Requirements:
    pip install 'imagetrust[desktop]'
    pip install pyinstaller
"""

import os
from pathlib import Path

# Project root (where this spec file is located)
PROJECT_ROOT = Path(SPECPATH)

block_cipher = None

# Data files to include
datas = [
    # Configuration files
    (str(PROJECT_ROOT / 'configs'), 'configs'),
]

# Add assets folder if it exists
assets_path = PROJECT_ROOT / 'assets'
if assets_path.exists():
    datas.append((str(assets_path), 'assets'))

# Hidden imports - modules PyInstaller might miss
hiddenimports = [
    # Core dependencies
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'numpy',
    'scipy',
    'scipy.ndimage',
    'sklearn',
    'sklearn.linear_model',
    'sklearn.isotonic',
    'sklearn.calibration',

    # Deep learning (conditionally loaded)
    'torch',
    'torchvision',
    'torchvision.transforms',
    'transformers',
    'timm',
    'timm.models',

    # Qt/PySide6
    'PySide6',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',

    # ImageTrust modules
    'imagetrust',
    'imagetrust.core',
    'imagetrust.core.config',
    'imagetrust.core.types',
    'imagetrust.detection',
    'imagetrust.detection.multi_detector',
    'imagetrust.detection.calibration',
    'imagetrust.detection.generator_identifier',
    'imagetrust.utils',
    'imagetrust.utils.scoring',
    'imagetrust.utils.helpers',
    'imagetrust.utils.image_utils',
    'imagetrust.utils.logging',
    'imagetrust.metadata',
    'imagetrust.metadata.exif_parser',
    'imagetrust.desktop',
    'imagetrust.desktop.app',

    # Optional dependencies
    'exifread',
    'cv2',
]

# Modules to exclude (reduce bundle size)
excludes = [
    # Development tools
    'pytest',
    'unittest',
    'test',
    'tests',
    'sphinx',
    'jupyter',
    'notebook',
    'IPython',

    # Not needed for desktop
    'streamlit',
    'fastapi',
    'uvicorn',
    'starlette',

    # Tkinter (using PySide6)
    'tkinter',
    '_tkinter',

    # Large optional packages
    'matplotlib',
    'pandas',
]

a = Analysis(
    # Entry point: PySide6 desktop app
    [str(PROJECT_ROOT / 'src' / 'imagetrust' / 'desktop' / 'app.py')],

    pathex=[str(PROJECT_ROOT / 'src')],
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
    optimize=1,  # Basic optimization
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Executable configuration
exe = EXE(
    pyz,
    a.scripts,
    [],  # Empty for folder mode; a.binaries + a.datas for onefile
    exclude_binaries=True,  # True for folder mode
    name='ImageTrust',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        'vcruntime140.dll',
        'python*.dll',
        'Qt*.dll',
        'PySide6*.dll',
    ],
    runtime_tmpdir=None,
    console=False,  # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Icon (create assets/icon.ico for custom icon)
    icon=str(assets_path / 'icon.ico') if (assets_path / 'icon.ico').exists() else None,
)

# Collect all files into distribution folder
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        'vcruntime140.dll',
        'python*.dll',
        'Qt*.dll',
    ],
    name='ImageTrust',
)
