"""
modules/fft.py  —  Calibrated FFT analysis (v2)
"""

import io
import math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pil_to_gray(pil_image: Image.Image) -> np.ndarray:
    return np.array(pil_image.convert("L"), dtype=np.float32)


def _compute_spectrum(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(gray)
    return np.log1p(np.abs(np.fft.fftshift(f)))


def _spectrum_to_pil(spectrum: np.ndarray) -> Image.Image:
    norm = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0d1117")
    ax.imshow(norm, cmap="inferno", interpolation="nearest")
    ax.set_title("FFT Frequency Spectrum", color="#e6edf3", fontsize=11, pad=8)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _grid_artifact_score(spectrum: np.ndarray) -> float:
    """
    Detect periodic grid artifacts.
    Uses fraction of pixels above 4σ vs expected Gaussian tail.
    AI upsampling creates regular bright spots far from the mean.
    """
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    mask = np.ones_like(spectrum, dtype=bool)
    mask[cy - 20:cy + 20, cx - 20:cx + 20] = False
    outer = spectrum[mask]

    mean_v, std_v = outer.mean(), outer.std()
    threshold = mean_v + 4.0 * std_v

    actual_frac   = (outer > threshold).mean()   # fraction of pixels above threshold
    expected_frac = 0.0001                        # ~Gaussian 4σ tail ≈ 0.003%, be lenient

    # excess = how many times more than expected
    excess = actual_frac / (expected_frac + 1e-10)
    # Score: excess of 1x → 0, excess of 50x → 1.0
    return float(np.clip((excess - 1.0) / 49.0, 0.0, 1.0))


def _high_freq_ratio(spectrum: np.ndarray) -> float:
    """
    Ratio of mid-high frequency energy vs low frequency energy.
    Normalized so real photos score ~0.0–0.35, AI images tend higher.
    """
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    r_low  = min(h, w) // 10
    r_mid  = min(h, w) // 4

    low_energy  = spectrum[dist < r_low].sum()
    mid_energy  = spectrum[(dist >= r_low) & (dist < r_mid)].sum()

    ratio = mid_energy / (low_energy + 1e-9)
    # Empirical: natural photos ~0.5–1.5, AI-gen often 2.0+
    return float(np.clip((ratio - 0.5) / 2.5, 0.0, 1.0))


def _radial_profile(spectrum: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """Average spectrum energy along radial shells — used for variance check."""
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).ravel()
    vals = spectrum.ravel()
    max_r = min(cy, cx)
    bins = np.linspace(0, max_r, n_bins + 1)
    profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (dist >= bins[i]) & (dist < bins[i + 1])
        if mask.any():
            profile[i] = vals[mask].mean()
    return profile


def _power_law_deviation(spectrum: np.ndarray) -> float:
    """
    Real photos follow a 1/f^α power spectrum.
    Measure deviation from a fitted power law — AI images deviate more.
    """
    profile = _radial_profile(spectrum)
    # Fit log-log line (skip DC)
    x = np.arange(2, len(profile))
    y = profile[2:]
    # Only fit where y > 0
    valid = y > 0
    if valid.sum() < 10:
        return 0.0
    log_x = np.log(x[valid])
    log_y = np.log(y[valid])
    try:
        coeffs = np.polyfit(log_x, log_y, 1)
        residuals = log_y - np.polyval(coeffs, log_x)
        deviation = float(np.std(residuals))
    except Exception:
        return 0.0
    # Real photos: residual std ~0.1–0.4. AI images: can be 0.5+
    return float(np.clip((deviation - 0.1) / 0.6, 0.0, 1.0))


def analyze(pil_image: Image.Image) -> dict:
    gray     = _pil_to_gray(pil_image)
    spectrum = _compute_spectrum(gray)
    spec_img = _spectrum_to_pil(spectrum)

    grid  = _grid_artifact_score(spectrum)
    hf    = _high_freq_ratio(spectrum)
    pwdev = _power_law_deviation(spectrum)

    # Weighted combo — power law is most principled signal
    score = float(np.clip(0.35 * grid + 0.30 * hf + 0.35 * pwdev, 0.0, 1.0))

    flags = []
    if grid > 0.50:
        flags.append("Periodic grid artifacts detected in FFT spectrum")
    if hf > 0.55:
        flags.append("Unusual high-frequency energy distribution")
    if pwdev > 0.55:
        flags.append("Spectrum deviates from natural 1/f power law")
    if score < 0.25:
        flags.append("FFT pattern consistent with natural photography")

    return {
        "spectrum_image":   spec_img,
        "grid_score":       round(grid, 3),
        "high_freq_ratio":  round(hf, 3),
        "power_dev":        round(pwdev, 3),
        "score":            round(score, 3),
        "flags":            flags,
    }
