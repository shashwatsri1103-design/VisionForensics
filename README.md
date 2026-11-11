# VisionForensics AI

A modular, explainable image forensics tool that estimates whether an uploaded image is a **real photograph** or an **AI-generated image**.

---

## Features

| Module | What it checks |
|---|---|
| **Metadata** | EXIF camera model, lens, date, software, GPS |
| **FFT Analysis** | Frequency spectrum grid artifacts, high-frequency energy |
| **Noise Analysis** | Sensor noise sigma, smooth region detection |
| **JPEG Analysis** | Compression quality, bytes/pixel, blocking artifacts |
| **Face Analysis** | Eye symmetry, ear alignment, face proportions via MediaPipe |
| **Final Report** | Weighted score → AI/Real verdict + confidence + evidence list |

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

## Build a Windows executable

On Windows, install Python 3.12, then double-click `build_exe.bat`. The build installs the app dependencies and PyInstaller in a local virtual environment. The executable will be created at `dist\VisionForensics\VisionForensics.exe`; keep the entire `dist\VisionForensics` folder together when moving or sharing the app.

The first launch needs an internet connection to download the AI classifier models. The resulting analysis app runs locally and opens in your browser at **http://localhost:8501**.

## Deploy a shareable web app

This project can be deployed from GitHub with Streamlit Community Cloud:

1. Push this project folder to a GitHub repository. Keep `app.py`, `requirements.txt`, and `modules/` at the repository root.
2. In Streamlit Community Cloud, create an app from that repository and select `app.py` as the entry point.
3. Share the public app URL provided by Streamlit.

Uploaded images are sent to the hosting server for analysis. The app is designed to analyze them in memory, but a public deployment is not local-only processing.

---

## Project Structure

```
VisionForensics/
├── app.py                  # Main Streamlit UI
├── requirements.txt
├── README.md
└── modules/
    ├── __init__.py
    ├── metadata.py         # EXIF extraction & analysis
    ├── fft.py              # Fast Fourier Transform analysis
    ├── noise.py            # Sensor noise estimation
    ├── jpeg.py             # Compression artifact analysis
    ├── face.py             # MediaPipe face mesh & geometry
    └── report.py           # Score aggregation & report generation
```

---

## How It Works

Each module independently analyzes the image and returns:
- A suspicion **score** (0 = natural, 1 = highly suspicious)
- A list of **flags** (human-readable evidence strings)

The `report.py` module combines scores with calibrated weights:

| Module   | Weight |
|----------|--------|
| Metadata | 30%    |
| Noise    | 25%    |
| FFT      | 20%    |
| Face     | 15%    |
| JPEG     | 10%    |

---

## Limitations & Disclaimer

- This is a **heuristic MVP** — not a certified forensic tool.
- Results are **probabilistic estimates**, not definitive proof.
- Does **not** detect Google SynthID or C2PA cryptographic watermarks.
- False positives (real images flagged as AI) and false negatives can occur.

---

## Future Improvements (Placeholders)

- [ ] Trained AI image classifier (ResNet / ViT fine-tuned on CIFAKE dataset)
- [ ] C2PA / Content Credentials verification
- [ ] OCR text consistency analysis
- [ ] Lighting direction & shadow coherence
- [ ] Object edge consistency map
- [ ] Improved DeepFake detection (FaceForensics++ model)

---

*Built with Python · Streamlit · OpenCV · MediaPipe · NumPy · Pillow*
