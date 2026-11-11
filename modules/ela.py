"""
modules/ela.py
Error Level Analysis (ELA) Module
Detects uniform compression levels typical of raw AI generation.
"""
import io
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze(pil_image: Image.Image) -> dict:
    """
    Performs Error Level Analysis by resaving the image at 90% quality 
    and analyzing the differences.
    """
    original = pil_image.convert("RGB")
    
    # 1. Resave at 90% quality
    temp = io.BytesIO()
    original.save(temp, "JPEG", quality=90)
    temp.seek(0)
    resaved = Image.open(temp)
    
    # 2. Compute absolute difference
    ela_image = ImageChops.difference(original, resaved)
    
    # 3. Enhance brightness for visualization
    extrema = ela_image.getextrema()
    # extrema is a tuple of (min, max) for each band
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    ela_display = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    # 4. Compute metrics on the grayscale difference
    ela_array = np.array(ela_image.convert("L"), dtype=np.float32)
    mean_val = float(np.mean(ela_array))
    std_val  = float(np.std(ela_array))
    
    # AI generators typically output images with uniform compression histories.
    # Real images (especially edited or saved multiple times) show varied ELA blocks.
    # Therefore, extremely low std_val (very uniform difference) is suspicious.
    score = 0.0
    flags = []
    
    if std_val < 3.0:
        score += 0.50
        flags.append(f"ELA shows uniformly low compression variance (std={std_val:.1f}) — typical of raw AI output")
    elif std_val < 6.0:
        score += 0.25
        flags.append(f"ELA shows moderate uniform compression (std={std_val:.1f})")
    elif std_val > 15.0:
        flags.append(f"ELA variance is high (std={std_val:.1f}) — typical of real/edited photos")
        
    # Render ELA Map for UI
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0d1117")
    im = ax.imshow(np.array(ela_display.convert("L")), cmap="magma", interpolation="nearest")
    ax.set_title("Error Level Analysis Map", color="#e6edf3", fontsize=11, pad=8)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    ela_map = Image.open(buf).copy()

    return {
        "ela_map": ela_map,
        "mean_diff": round(mean_val, 2),
        "std_diff": round(std_val, 2),
        "score": round(float(np.clip(score, 0.0, 1.0)), 3),
        "flags": flags
    }
