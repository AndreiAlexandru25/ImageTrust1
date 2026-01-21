# ImageTrust Assets

This folder contains branding assets for the ImageTrust desktop application.

## Required Files for Windows .exe

### Icon (`icon.ico`)

Create a Windows icon file with the following sizes embedded:
- 16x16
- 32x32
- 48x48
- 64x64
- 128x128
- 256x256

**Tools to create .ico files:**
- [IcoConvert](https://icoconvert.com/) (online)
- [GIMP](https://www.gimp.org/) (with ICO plugin)
- [ImageMagick](https://imagemagick.org/):
  ```bash
  convert logo.png -define icon:auto-resize=256,128,64,48,32,16 icon.ico
  ```

### Recommended Icon Design

The ImageTrust icon should convey:
- **Trust/Security**: Shield, checkmark, or lock motif
- **Image/Visual**: Camera lens, frame, or pixel pattern
- **AI/Technology**: Circuit patterns, neural network nodes

**Color palette:**
- Primary: `#6366f1` (Indigo/purple - matches app accent)
- Secondary: `#22c55e` (Green - for "verified/real")
- Accent: `#ef4444` (Red - for "AI detected")
- Background: `#0f1117` (Dark)

## File Structure

```
assets/
├── README.md           # This file
├── icon.ico            # Windows application icon
├── icon.png            # High-res PNG version (1024x1024)
├── icon-16.png         # Taskbar icon
├── icon-32.png         # Small icon
├── icon-256.png        # Large icon
├── logo.svg            # Vector logo (scalable)
├── splash.png          # Splash screen (optional, 600x400)
└── banner.png          # README/documentation banner (optional)
```

## Usage in PyInstaller

The `ImageTrust.spec` file automatically looks for `assets/icon.ico`:

```python
exe = EXE(
    ...
    icon=str(PROJECT_ROOT / 'assets' / 'icon.ico'),
)
```

## Placeholder Icon

Until a custom icon is created, the application will use the default Python/Qt icon.

To test with a placeholder:
1. Download any free icon from [Flaticon](https://www.flaticon.com/) or [Icons8](https://icons8.com/)
2. Convert to ICO format
3. Save as `assets/icon.ico`

## License

Any custom icons should be:
- Created originally, OR
- Licensed for commercial use (check licenses for downloaded assets)

For thesis submissions, ensure all assets are properly attributed.
