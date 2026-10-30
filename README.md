# ImageTruth Analyzer

AI-based image recognition system that identifies whether an image is **real or AI-generated**, provides **confidence scores**, and detects **manipulation** (editing, compression, resizing).

## Features

- **AI vs Real Classification** with probability scores (e.g., AI-generated: 96%, Real: 4%)
- **Manipulation Detection** using Error Level Analysis (ELA) - detects editing, heavy compression, compositing
- **Forensic Indicators** - frequency spectrum analysis, noise fingerprinting, DCT artifact analysis, texture uniformity checks
- **Clean web interface** - upload an image and instantly receive an authenticity report

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python 3.12 / 3.13 / 3.14 + Flask
- **Image Processing:** Pillow, NumPy (no TensorFlow, no heavy ML frameworks)

## How to Run in VS Code

1. **Open the project folder** in VS Code:
   - `File > Open Folder` and select this project folder

2. **Create a virtual environment** (open the VS Code terminal / PowerShell):
   ```
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```
   python train_model.py

python app.py
   ```

5. **Open your browser** and visit:
   ```
   http://127.0.0.1:5000
   ```

## How It Works

1. **Error Level Analysis (ELA)** - Re-compresses the image at a known quality level and compares differences. High error levels indicate editing or compression.
2. **Frequency Spectrum Analysis** - Analyzes FFT energy distribution to detect GAN/AI-specific spectral signatures.
3. **Noise Fingerprinting** - Inspects high-frequency noise consistency; AI-generated images often have characteristically smooth or uniform noise.
4. **DCT Artifact Analysis** - Looks for periodic 8x8 block patterns typical of JPEG and AI upscaling artifacts.
5. **Texture Uniformity** - Measures entropy and edge sharpness to spot unnaturally smooth or repeating texture fields typical of AI generators.

## Project Structure

```
mini project/
├── app.py               <-- Flask backend (entry point)
├── analyzer.py          <-- image forensic analysis engine
├── requirements.txt
├── templates/
│   └── index.html       <-- frontend page
└── static/
    ├── style.css        <-- styling
    └── uploads/         <-- uploaded images (auto-cleaned)
```

## Notes

- Supports: PNG, JPG, JPEG, BMP, WEBP, TIFF, GIF (max 16 MB)
- No TensorFlow / PyTorch required - the bundled classifier is trained with NumPy + Pillow
- Works on Python 3.12, 3.13, and 3.14
## Retraining the model

If you replace the images in `dataset/real/` and `dataset/ai_generated/`, run `python train_model.py` once. It recreates `models/ai_classifier.npz`, which the backend loads automatically.
