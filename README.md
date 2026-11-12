# VisionForensics AI

A modular, explainable image forensics tool that estimates whether an uploaded image is a **real photograph**, an **AI-generated image**, or a **deepfake/manipulated image**.

---

## Features

| Module | What it checks |
|---|---|
| **Pretrained classifier** | AI-generated, deepfake/manipulated, and real class probabilities |
| **Metadata** | EXIF camera model, lens, date, software, GPS |
| **FFT Analysis** | Frequency spectrum grid artifacts, high-frequency energy |
| **Noise Analysis** | Sensor noise sigma, smooth region detection |
| **JPEG Analysis** | Compression quality, bytes/pixel, blocking artifacts |
| **Face Analysis** | Eye symmetry, ear alignment, face proportions via MediaPipe |
| **Final Report** | AI/Deepfake/Real verdict, confidence, and evidence list |

The pretrained model is `prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0`. On first use, the app downloads its weights; this may take a few minutes and requires an internet connection.

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

The app opens automatically at **http://localhost:8501**.

## Build a Windows executable

On Windows, install Python 3.12, then double-click `build_exe.bat`. The build installs the app dependencies and PyInstaller in a local virtual environment. The executable will be created at `dist\VisionForensics\VisionForensics.exe`; keep the entire `dist\VisionForensics` folder together when moving or sharing the app.

The first launch needs an internet connection to download the three-class classifier. The app runs locally and opens in your browser at **http://localhost:8501**.

## Deploy a shareable web app

This project can be deployed from GitHub with Streamlit Community Cloud:

1. Push this project folder to a GitHub repository. Keep `app.py`, `requirements.txt`, and `modules/` at the repository root.
2. In Streamlit Community Cloud, create an app from that repository and select `app.py` as the entry point.
3. Share the public app URL provided by Streamlit.

Uploaded images are sent to the hosting server for analysis. The app analyzes them in memory, but a public deployment is not local-only processing.

---

## Project Structure

```
VisionForensics/
├── app.py                  # Main Streamlit UI
├── requirements.txt
├── README.md
└── modules/
    ├── ai_classifier.py   # Three-class pretrained classifier
    ├── metadata.py        # EXIF extraction & analysis
    ├── fft.py             # Fast Fourier Transform analysis
    ├── noise.py           # Sensor noise estimation
    ├── jpeg.py            # Compression artifact analysis
    ├── face.py            # MediaPipe face mesh & geometry
    ├── watermark.py       # Limited visual watermark check
    ├── ela.py             # Error level analysis
    └── report.py          # Score aggregation & report generation
```

---

## How It Works

Forensic modules independently analyze the image and return a suspicion **score** (0 = natural, 1 = suspicious) plus human-readable evidence flags. The pretrained classifier separately reports probabilities for AI-generated, deepfake/manipulated, and real classes.

When the classifier is available, it contributes 80% of the AI-generation score. Remaining base weights are metadata 5%, ELA 3%, noise 5%, FFT 3%, JPEG 2%, and face 2%. Without the classifier, the app uses the heuristic fallback weights below. A detected visual watermark may add up to 15 percentage points (2 when possible post-processing signals are present); this does not authenticate the watermark origin.

| Heuristic signal | Model available | Fallback |
|---|---:|---:|
| Metadata | 5% | 20% |
| ELA + noise | 8% | 35% |
| FFT | 3% | 18% |
| JPEG + face | 4% | 27% |

---

## Limitations & Disclaimer

- This is an analysis aid, not a certified forensic tool.
- Results are probabilistic estimates, not definitive proof.
- Does not authenticate Google SynthID or C2PA cryptographic credentials.
- False positives and false negatives can occur. The hosted model has not been validated against your own labeled image set.

---

## Future Improvements

- [ ] Evaluate and calibrate the classifier using a held-out labeled dataset
- [ ] C2PA / Content Credentials verification
- [ ] OCR text consistency analysis
- [ ] Lighting direction & shadow coherence
- [ ] Object edge consistency map

*Built with Python · Streamlit · OpenCV · MediaPipe · NumPy · Pillow*
