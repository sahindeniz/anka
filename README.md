# 🔭 Astro Maestro Pro (Anka)

**Astrophotography image processing application** built with Python & PyQt6.

Process deep-sky images with professional-grade tools — background extraction, star reduction, deconvolution, noise reduction, histogram stretching, color calibration, and more.
 
Turkish manual: [KULLANIM_KILAVUZU.md](./KULLANIM_KILAVUZU.md)

---

## ✨ Features

| Category | Tools |
|----------|-------|
| **Image I/O** | FITS, TIFF, PNG, JPG support · Multi-image filmstrip viewer · Drag & drop |
| **Stretching** | Photoshop-style Histogram Editor (Levels, Curves, Adjustments) · VeraLux HMS Auto Stretch · STF (Screen Transfer Function) |
| **Background** | Gradient extraction (Polynomial / RBF / AI) · GraXpert integration |
| **Stars** | Star Smaller (natural star reduction) · Star Aberration Remover · Star Recomposition · StarNet++ / StarXterminator bridge |
| **Enhancement** | Deconvolution (Richardson-Lucy / Wiener) · Sharpening · Nebula Enhancement · Morphology |
| **Color** | Color Calibration · White Balance · Vibrance / Saturation / Hue |
| **Noise** | Multi-algorithm noise reduction (Wavelet / Bilateral / NLM) · NoiseXterminator bridge |
| **Astrometry** | ASTAP Plate Solving · WCS annotation overlay |
| **Stacking** | DSS-style stacking with Bias / Dark / Flat calibration |
| **Scripting** | Built-in Python Script Editor with syntax highlighting |
| **Other** | Undo / Redo history · R/G/B/L channel viewer · Draggable toolbar · Auto pipeline |

---

## 🚀 Quick Start

### Windows (Recommended)

1. **Download** — Clone or download this repository:
   ```
   git clone https://github.com/sahindeniz/anka.git
   cd anka
   ```

2. **Install** — Double-click `install.bat` or run:
   ```
   install.bat
   ```
   This will check Python, install all dependencies, and create a desktop shortcut.

3. **Run** — Double-click `setup_and_run.bat` or:
   ```
   python main.py
   ```

### ZIP Download

- Fastest option: click `Code > Download ZIP` on GitHub.
- Automatic package ZIP: open the `Actions` tab and download the latest `astro-maestro-pro-zip` artifact from the `Build ZIP package` workflow.
- Versioned ZIPs: when a `v*` tag is pushed, the same ZIP is attached to the matching GitHub `Release`.

### Manual Install (Windows / macOS / Linux)

1. **Prerequisites:** Python 3.9 or later

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run:**
   ```bash
   python main.py
   ```

---

## 📦 Requirements

```
numpy
opencv-python
astropy
scikit-image
scipy
matplotlib
PyQt6
astroquery
pyyaml
tifffile
PyWavelets
```

### Optional (for advanced features)

| Package | Feature |
|---------|---------|
| [ASTAP](https://www.ling.net/astap/) | Plate Solving (set path in Settings → ASTAP) |
| [GraXpert](https://www.graxpert.com/) | AI Background Extraction |
| [StarNet++](https://www.starnetastro.com/) | AI Star Removal |
| [NoiseXterminator](https://www.rc-astro.com/) | AI Noise Reduction |

---

## 🖥️ Screenshots

*Load your deep-sky FITS/TIFF images → Process with built-in tools → Export results*

The application features a dark-themed professional interface with:
- Central image viewer with zoom/pan and multi-image filmstrip
- Right-side Histogram Editor (Levels, Curves, Adjustments)
- Flyout process panels accessible from the toolbar
- Real-time live preview

---

## 📁 Project Structure

```
anka/
├── main.py                  # Entry point
├── install.bat              # Windows installer
├── setup_and_run.bat        # Windows launcher
├── requirements.txt         # Python dependencies
├── core/
│   ├── loader.py            # FITS/TIFF/PNG image loader
│   └── version.py           # Version & update checker
├── gui/
│   ├── app.py               # Main application window
│   ├── canvas.py            # Image canvas widget
│   ├── histogram_editor.py  # Photoshop-style histogram editor
│   ├── panels.py            # Process parameter panels
│   ├── recomposition.py     # Star recomposition dialog
│   ├── script_editor.py     # Python script editor
│   └── ...
├── processing/
│   ├── background.py        # Background extraction
│   ├── starsmaller.py       # Star reduction
│   ├── deconvolution.py     # Deconvolution algorithms
│   ├── noise_reduction.py   # Noise reduction
│   ├── color_calibration.py # Color calibration
│   ├── sharpening.py        # Sharpening
│   ├── stretch.py           # Histogram stretch
│   └── ...
├── ai/
│   ├── astap_bridge.py      # ASTAP plate solving bridge
│   ├── starnet_bridge.py    # StarNet++ bridge
│   └── ...
├── astrometry/
│   ├── plate_solver.py      # Plate solving logic
│   └── wcs_annotator.py     # WCS overlay
└── analysis/
    └── statistics.py        # Image statistics
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs via [Issues](https://github.com/sahindeniz/anka/issues)
- Submit feature requests
- Open pull requests

---

## 📄 License

This project is open source. See the repository for license details.

---

## 👤 Author

**Deniz Sahin** — [@sahindeniz](https://github.com/sahindeniz)

---

*Built with ❤️ for the astrophotography community*
