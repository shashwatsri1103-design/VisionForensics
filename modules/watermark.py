"""
modules/watermark.py  v5  — Dual-method detection

Method 1 (Primary): Isolation-based blob detection
  - Find very bright blobs in corners
  - Require the blob to be isolated (≥ 50 gray units above surroundings)
  - Check star shape (solidity + extent)

Method 2 (Backup): Template matching
  - Generate synthetic 4-pointed star kernels
  - Run multi-scale normalised cross-correlation at threshold 0.82
  - Catches watermarks that are embedded on uniform/white backgrounds

A watermark is flagged if EITHER method passes.
"""
import cv2
import numpy as np
from PIL import Image

# ─── Tuning ──────────────────────────────────────────────────────────────────
BRIGHT_THRESH   = 230       # brightness threshold for blob detection
MIN_AREA_FRAC   = 0.0004    # min blob area as fraction of corner patch
MAX_AREA_FRAC   = 0.10      # max blob area
MIN_ASPECT      = 0.45      # min aspect ratio (roughly square)
MIN_ISOLATION   = 50        # gray units blob must exceed neighbourhood
NEIGH_EXPAND    = 3         # neighbourhood expand factor around blob bbox
TMPL_THRESHOLD  = 0.72
TMPL_MARGIN     = 0.18
TMPL_SCALE_FRACS = (0.28, 0.32, 0.36, 0.40)
# ─────────────────────────────────────────────────────────────────────────────


def _star_kernel(size: int) -> np.ndarray:
    """Filled four-point astroid used by translucent Gemini sparkles."""
    k = np.zeros((size, size), dtype=np.uint8)
    c = (size - 1) / 2.0
    for y in range(size):
        for x in range(size):
            nx = (x - c) / c
            ny = (y - c) / c
            if abs(nx) ** (2 / 3) + abs(ny) ** (2 / 3) <= 1.0:
                k[y, x] = 255
    return k


def _is_star_shaped(mask: np.ndarray) -> bool:
    """4-pointed star: low solidity AND low extent."""
    m = cv2.moments(mask)
    if m["m00"] == 0:
        return False
    area = m["m00"]
    pts  = cv2.findNonZero(mask)
    if pts is None:
        return False
    hull      = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)
    if hull_area < 1:
        return False
    solidity = area / hull_area
    x, y, bw, bh = cv2.boundingRect(pts)
    extent = area / (bw * bh + 1e-6)
    return (0.28 <= solidity <= 0.84) and (0.18 <= extent <= 0.78)


# ── Method 1: isolation-based blob ──────────────────────────────────────────

def _isolation_check(patch: np.ndarray) -> tuple[bool, float]:
    ph, pw = patch.shape
    _, binary = cv2.threshold(patch, BRIGHT_THRESH, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bx   = int(stats[i, cv2.CC_STAT_LEFT])
        by   = int(stats[i, cv2.CC_STAT_TOP])
        bw   = int(stats[i, cv2.CC_STAT_WIDTH])
        bh   = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx   = int(centroids[i][0])
        cy   = int(centroids[i][1])

        min_a = max(12, int(ph * pw * MIN_AREA_FRAC))
        max_a = int(ph * pw * MAX_AREA_FRAC)
        if not (min_a <= area <= max_a):
            continue
        if bw == 0 or bh == 0:
            continue
        if min(bw, bh) / max(bw, bh) < MIN_ASPECT:
            continue

        blob_mean = float(np.mean(patch[labels == i]))
        exp = max(bw, bh) * NEIGH_EXPAND
        nx1, nx2 = max(0, cx - exp), min(pw, cx + exp)
        ny1, ny2 = max(0, cy - exp), min(ph, cy + exp)
        nb_region = patch[ny1:ny2, nx1:nx2]
        nb_mask   = labels[ny1:ny2, nx1:nx2]
        bg_pixels = nb_region[nb_mask != i]
        if bg_pixels.size == 0:
            continue
        contrast = blob_mean - float(np.mean(bg_pixels))
        if contrast < MIN_ISOLATION:
            continue

        crop = (labels[by:by+bh, bx:bx+bw] == i).astype(np.uint8)
        if _is_star_shaped(crop):
            conf = min(1.0, contrast / 160.0)
            return True, conf

    return False, 0.0


# ── Method 2: template matching ─────────────────────────────────────────────

def _template_check(patch: np.ndarray) -> tuple[bool, float]:
    gray_f = patch.astype(np.float32)
    best   = 0.0
    min_dim = min(patch.shape)
    scales = {max(14, int(min_dim * frac)) for frac in TMPL_SCALE_FRACS}
    for size in scales:
        if size > patch.shape[0] or size > patch.shape[1]:
            continue
        tmpl = _star_kernel(size).astype(np.float32)
        res  = cv2.matchTemplate(gray_f, tmpl, cv2.TM_CCOEFF_NORMED)
        _, mv, _, _ = cv2.minMaxLoc(res)
        if mv > best:
            best = mv
    return best >= TMPL_THRESHOLD, best


# ── Public API ───────────────────────────────────────────────────────────────

def detect_sparkle(pil_image: Image.Image) -> dict:
    """Detect Gemini-style 4-pointed star watermarks in image corners."""
    result = dict(detected=False, location=None, confidence=0.0, score=0.0, flags=[])

    img  = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    frac_h = max(int(h * 0.22), 70)
    frac_w = max(int(w * 0.22), 70)

    corners = {
        "top-left":     gray[:frac_h, :frac_w],
        "top-right":    gray[:frac_h,  w - frac_w:],
        "bottom-left":  gray[h - frac_h:, :frac_w],
        "bottom-right": gray[h - frac_h:, w - frac_w:],
    }

    best_conf  = 0.0
    best_label = None
    best_method = None
    template_scores = {}

    for label, patch in corners.items():
        if patch.size == 0:
            continue

        # Method 1: Isolation
        found1, c1 = _isolation_check(patch)
        if found1 and c1 > best_conf:
            best_conf, best_label, best_method = c1, label, "isolation"

        # Method 2: compare the strongest corner match against the others.
        _, template_scores[label] = _template_check(patch)

    if best_method != "isolation" and template_scores:
        ranked = sorted(template_scores.items(), key=lambda item: item[1], reverse=True)
        best_label, template_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if (template_score >= TMPL_THRESHOLD
                and template_score - second_score >= TMPL_MARGIN):
            best_conf = template_score
            best_method = "template"
        else:
            best_conf, best_label, best_method = 0.0, None, None

    if best_conf > 0:
        result.update(
            detected   = True,
            location   = best_label,
            confidence = round(best_conf, 3),
            score      = round(best_conf, 3),
        )
        result["flags"].append(
            f"✦ Gemini-style 4-pointed star watermark detected at {best_label} "
            f"(confidence {best_conf:.2f}, method: {best_method})"
        )
    else:
        result["flags"].append("No isolated bright star found in corners — no visual watermark")

    return result
