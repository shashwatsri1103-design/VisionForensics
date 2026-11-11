"""
modules/noise.py  — v3
Uses Poisson shot-noise correlation to separate real photos from AI images.

Key insight:
  Real camera sensors produce SHOT NOISE where variance ∝ local brightness.
  AI diffusion models produce noise that has NO such correlation (it's either
  absent after denoising, or is synthesis artifact noise with different statistics).

We test this with the "Poisson Correlation Score":
  1. Compute noise residual (high-pass filter)
  2. Divide image into blocks
  3. Measure local noise variance and local mean brightness per block
  4. Compute Pearson r between variance and brightness
  5. Real photos → r > 0.45 (shot noise dominant)
     AI images  → r < 0.20 (flat / uncorrelated)
"""
import numpy as np
import cv2
import io
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _noise_residual(gray: np.ndarray) -> np.ndarray:
    """Subtract a Gaussian-blurred version to get the high-frequency noise residual."""
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    return (gray.astype(np.float32) - blur.astype(np.float32))


def _estimate_sigma(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    kernel   = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], dtype=np.float64)
    filtered = cv2.filter2D(gray.astype(np.float64), -1, kernel)
    return float(np.sqrt(np.pi / 2) * np.mean(np.abs(filtered)) / 6.0)


def _poisson_correlation(gray: np.ndarray, block: int = 24) -> float:
    """
    Returns Pearson r between local noise variance and local brightness.
    Higher r → more shot-noise-like → more likely real photo.
    """
    residual = _noise_residual(gray)
    h, w = gray.shape
    means, variances = [], []
    for r in range(0, h - block, block):
        for c in range(0, w - block, block):
            patch    = gray[r:r+block, c:c+block].astype(np.float32)
            res_p    = residual[r:r+block, c:c+block]
            lm  = float(np.mean(patch))
            lv  = float(np.var(res_p))
            means.append(lm)
            variances.append(lv)
    if len(means) < 10:
        return 0.0
    m_arr = np.array(means,    dtype=np.float32)
    v_arr = np.array(variances,dtype=np.float32)
    corr  = np.corrcoef(m_arr, v_arr)[0, 1]
    return float(np.clip(corr, -1.0, 1.0))


def _local_noise_map(gray: np.ndarray, block: int = 32) -> np.ndarray:
    h, w = gray.shape
    rows, cols = h // block, w // block
    nm = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            b = gray[r*block:(r+1)*block, c*block:(c+1)*block].astype(np.float64)
            nm[r, c] = _estimate_sigma(b)
    return nm


def _map_to_pil(nm: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0d1117")
    im = ax.imshow(nm, cmap="plasma", interpolation="nearest")
    ax.set_title("Local Noise Map", color="#e6edf3", fontsize=11, pad=8)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.label.set_color("#e6edf3")
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig); buf.seek(0)
    return Image.open(buf).copy()


def analyze(pil_image: Image.Image) -> dict:
    gray  = np.array(pil_image.convert("L"), dtype=np.float32)
    sigma = _estimate_sigma(gray)
    block = max(8, min(gray.shape) // 20)
    nm    = _local_noise_map(gray, block)
    nmap  = _map_to_pil(nm)
    r_shot = _poisson_correlation(gray)

    smooth_threshold = 1.5
    smooth_pct = float((nm < smooth_threshold).mean() * 100)
    noise_var  = float(nm.std())

    score = 0.0
    flags = []

    # ── Global sigma ──────────────────────────────────────────────────────
    if sigma < 0.5:
        score += 0.45
        flags.append(f"Extremely uniform image (σ={sigma:.2f}) — near-perfectly smooth")
    elif sigma < 1.0:
        score += 0.20
        flags.append(f"Very low sensor noise (σ={sigma:.2f}) — possible AI generation")
    elif sigma < 3.0:
        score += 0.05
        flags.append(f"Below-average sensor noise (σ={sigma:.2f}) — may be smartphone NR")
    else:
        # Positive evidence for real photo — reduce score slightly
        score -= 0.05
        flags.append(f"Natural sensor noise detected (σ={sigma:.2f}) — consistent with real camera")

    # ── Poisson shot-noise correlation ────────────────────────────────────
    # Real cameras: r > 0.45 (noise grows with brightness)
    # AI models:    r < 0.20 (no such relationship)
    if r_shot > 0.45:
        score -= 0.20
        flags.append(f"Shot-noise pattern detected (r={r_shot:.2f}) — characteristic of real camera sensor")
    elif r_shot > 0.25:
        flags.append(f"Weak shot-noise correlation (r={r_shot:.2f}) — ambiguous")
    elif r_shot < 0.10:
        score += 0.30
        flags.append(f"No shot-noise correlation (r={r_shot:.2f}) — absent in real sensors, suspicious")

    # ── Smooth region percentage ──────────────────────────────────────────
    if smooth_pct > 90:
        score += 0.25
        flags.append(f"{smooth_pct:.0f}% of blocks are unnaturally smooth")
    elif smooth_pct > 70:
        score += 0.10
        flags.append(f"{smooth_pct:.0f}% smooth regions (high but possible with aggressive NR)")

    if noise_var < 0.3:
        score += 0.10
        flags.append("Noise is spatially uniform — unnatural for camera sensors")

    return dict(
        global_sigma    = round(sigma, 3),
        shot_noise_corr = round(r_shot, 3),
        smoothness_pct  = round(smooth_pct, 1),
        noise_variance  = round(noise_var, 3),
        noise_map_img   = nmap,
        score           = round(float(np.clip(score, 0.0, 1.0)), 3),
        flags           = flags,
    )
