"""
modules/jpeg.py
Analyzes JPEG compression characteristics.
"""

import io
import numpy as np
from PIL import Image


def _estimate_jpeg_quality(pil_image: Image.Image) -> int:
    buf_ref = io.BytesIO()
    pil_image.save(buf_ref, format="JPEG", quality=95)
    ref_size = buf_ref.tell()
    lo, hi = 1, 95
    for _ in range(8):
        mid = (lo + hi) // 2
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=mid)
        if buf.tell() < ref_size * (mid / 95.0):
            lo = mid + 1
        else:
            hi = mid
    return (lo + hi) // 2


def _blocking_artifact_score(gray: np.ndarray) -> float:
    h, w = gray.shape
    h_edges, v_edges = [], []
    for col in range(8, w - 1, 8):
        diff = np.abs(gray[:, col].astype(float) - gray[:, col - 1].astype(float))
        h_edges.append(diff.mean())
    for row in range(8, h - 1, 8):
        diff = np.abs(gray[row, :].astype(float) - gray[row - 1, :].astype(float))
        v_edges.append(diff.mean())
    all_h = np.abs(np.diff(gray.astype(float), axis=1)).mean()
    all_v = np.abs(np.diff(gray.astype(float), axis=0)).mean()
    block_h = np.mean(h_edges) if h_edges else 0
    block_v = np.mean(v_edges) if v_edges else 0
    ratio = ((block_h / (all_h + 1e-6)) + (block_v / (all_v + 1e-6))) / 2.0
    return float(np.clip((ratio - 1.0) / 3.0, 0.0, 1.0))


def analyze(pil_image: Image.Image, file_size_bytes: int, file_format: str) -> dict:
    """
    Analyze JPEG characteristics of pil_image.
    Returns score 0-1 (higher = more AI-suspicious) and evidence flags.
    """
    w, h = pil_image.size
    gray = np.array(pil_image.convert("L"), dtype=np.float32)
    file_size_kb = file_size_bytes / 1024.0
    bpp = file_size_bytes / (w * h) if (w * h) > 0 else 0.0
    blocking = _blocking_artifact_score(gray)
    est_quality = _estimate_jpeg_quality(pil_image)
    score = 0.0
    flags = []

    fmt_upper = file_format.upper()
    if fmt_upper == "PNG":
        score += 0.20
        flags.append("PNG format — AI generators often output lossless PNG")
    elif fmt_upper == "WEBP":
        score += 0.10
        flags.append("WEBP format — common AI generator output format")

    if est_quality > 90:
        score += 0.05
        flags.append(f"Very high re-encode quality ({est_quality}) — minimal compression history")
    elif est_quality < 50 and blocking < 0.1:
        score += 0.20
        flags.append(f"Low quality ({est_quality}) with minimal JPEG blocking — possible re-encode")
    else:
        flags.append(f"Estimated compression quality: {est_quality}/95")

    if bpp < 0.05:
        score += 0.15
        flags.append(f"Very low file density ({bpp:.3f} B/px)")
    elif bpp > 3.0:
        score += 0.05
        flags.append(f"High file density ({bpp:.2f} B/px) — lossless or minimally compressed")
    else:
        flags.append(f"Normal file density ({bpp:.3f} B/px)")

    return {
        "estimated_quality": est_quality,
        "file_size_kb": round(file_size_kb, 1),
        "bytes_per_pixel": round(bpp, 4),
        "blocking_score": round(blocking, 3),
        "file_format": fmt_upper,
        "score": round(float(np.clip(score, 0.0, 1.0)), 3),
        "flags": flags,
    }
