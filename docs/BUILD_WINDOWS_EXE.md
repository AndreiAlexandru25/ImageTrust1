# Building Cyber Scout Windows EXE

This guide explains how to build the Cyber Scout desktop application as a standalone Windows executable.

## Overview

Cyber Scout is a native Windows desktop wrapper for the ImageTrust forensics UI. It uses:
- **pywebview** with WebView2 for the native window
- **Streamlit** for the web UI (runs locally)
- **PyInstaller** for bundling into an executable

## Prerequisites

### Required Software

1. **Python 3.10+** - Download from [python.org](https://python.org)
2. **Microsoft Edge WebView2 Runtime** - Required on target machines
   - Download: https://developer.microsoft.com/microsoft-edge/webview2/
   - Choose "Evergreen Bootstrapper" for automatic installation
   - Most Windows 10/11 machines already have this installed

### Python Dependencies

```powershell
# From project root with venv activated
pip install pyinstaller pywebview streamlit
```

## Build Commands

### Quick Build (One-Folder)

```powershell
# From project root
.\scripts\build_exe.ps1
```

Output: `dist\CyberScout\CyberScout.exe`

### Clean Build

```powershell
.\scripts\build_exe.ps1 -Clean
```

### One-File Build (Single EXE)

```powershell
.\scripts\build_exe.ps1 -OneFile
```

Output: `dist\CyberScout.exe` (larger file, slower startup)

### Manual Build

```powershell
# Using spec file
pyinstaller --noconfirm CyberScout.spec

# Or direct command
pyinstaller --noconsole --name CyberScout `
    --add-data "assets;assets" `
    --add-data "src\imagetrust\frontend;imagetrust\frontend" `
    --hidden-import streamlit `
    --hidden-import webview `
    src\imagetrust\frontend\desktop_launcher.py
```

## Output Structure

### One-Folder Build (Default)

```
dist/
└── CyberScout/
    ├── CyberScout.exe      # Main executable
    ├── assets/             # UI backgrounds
    │   └── ui/
    │       ├── landing_bg.png
    │       └── results_bg.png
    ├── imagetrust/         # Application code
    │   └── frontend/
    │       └── cyber_app.py
    └── [dependencies...]   # Python DLLs, etc.
```

### One-File Build

```
dist/
└── CyberScout.exe          # Single executable (all bundled)
```

## Testing the Build

1. **Double-click** `CyberScout.exe`
2. A native window should open (NOT a browser)
3. The Cyber Scout landing screen should appear
4. Upload an image and verify analysis works

### Expected Behavior

- Window title: "Cyber Scout - Image Forensics"
- No browser window opens
- No console window appears
- Clean shutdown when closing the window

## Troubleshooting

### WebView2 Not Found

**Error:** "Microsoft Edge WebView2 Runtime is not installed"

**Solution:**
1. Download WebView2 from Microsoft: https://developer.microsoft.com/microsoft-edge/webview2/
2. Install the "Evergreen Bootstrapper"
3. Restart the application

### Streamlit Server Timeout

**Error:** "Streamlit server did not start within 30 seconds"

**Possible causes:**
- Port 8501+ already in use
- Missing dependencies
- Antivirus blocking

**Solutions:**
- Close other applications using ports 8501-8600
- Disable antivirus temporarily
- Check Windows Firewall settings

### Missing Assets

**Error:** Black/missing backgrounds

**Solution:**
Verify assets are included:
```powershell
# Check dist folder
ls dist\CyberScout\assets\ui\
# Should show: landing_bg.png, results_bg.png
```

### Import Errors

**Error:** Module not found errors

**Solution:**
Add missing modules to `hiddenimports` in `CyberScout.spec`:
```python
hiddenimports = [
    'missing_module_name',
    # ...
]
```

Then rebuild.

### Window Doesn't Open

**Possible causes:**
- WebView2 not installed
- Python not finding Streamlit

**Debug:**
```powershell
# Run launcher directly to see errors
python src\imagetrust\frontend\desktop_launcher.py
```

## Distribution

### What to Include

To distribute Cyber Scout:

1. **For One-Folder build:**
   - Copy entire `dist\CyberScout\` folder
   - Users run `CyberScout.exe`

2. **For One-File build:**
   - Copy `dist\CyberScout.exe`
   - Single file, easier distribution

### Requirements for Target Machine

- Windows 10/11 (64-bit)
- Microsoft Edge WebView2 Runtime
- No Python installation required!

### Creating an Installer (Optional)

Use [Inno Setup](https://jrsoftware.org/isinfo.php) or [NSIS](https://nsis.sourceforge.io/) to create an installer:

```iss
; Example Inno Setup snippet
[Setup]
AppName=Cyber Scout
AppVersion=1.0.0
DefaultDirName={autopf}\CyberScout

[Files]
Source: "dist\CyberScout\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{commondesktop}\Cyber Scout"; Filename: "{app}\CyberScout.exe"
```

## CLI Alternative

If building fails, users can still run via CLI:

```powershell
# Requires Python environment
imagetrust desktop
```

Or launch Streamlit directly in browser:

```powershell
imagetrust cyber
```

## File Sizes

Expected sizes (approximate):
- One-folder build: ~200-500 MB (depending on included models)
- One-file build: ~150-400 MB compressed

To reduce size:
1. Exclude unused ML models in spec file
2. Use UPX compression (enabled by default)
3. Remove debug symbols

## Development

### Testing Changes Without Rebuilding

```powershell
# Run launcher directly
python src\imagetrust\frontend\desktop_launcher.py
```

### Modifying the Spec File

Key sections in `CyberScout.spec`:

- `datas` - Files to include (assets, frontend)
- `hiddenimports` - Modules PyInstaller misses
- `excludes` - Packages to skip (reduces size)
- `console=False` - Hide console window

## Support

If you encounter issues:

1. Check this troubleshooting guide
2. Run with debug: `.\scripts\build_exe.ps1 -Debug`
3. Check PyInstaller logs in `build\` folder
4. File an issue on GitHub with:
   - Error message
   - Python version
   - Windows version
   - Build command used
