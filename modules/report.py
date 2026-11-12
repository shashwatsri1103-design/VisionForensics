"""Combine image-analysis scores with a limited visual-watermark contribution."""

# Weights when ML classifier IS available
WEIGHTS_ML = {
    "ml":       0.80,
    "metadata": 0.05,
    "ela":      0.03,
    "noise":    0.05,
    "fft":      0.03,
    "jpeg":     0.02,
    "face":     0.02,
}

# Weights when ML classifier is NOT available (fallback)
WEIGHTS_HEURISTIC = {
    "metadata": 0.20,
    "ela":      0.15,
    "noise":    0.20,
    "fft":      0.18,
    "jpeg":     0.07,
    "face":     0.20,
}

WATERMARK_WEIGHT = 0.15
EDITED_WATERMARK_WEIGHT = 0.02
EDITING_SOFTWARE = (
    "photoshop", "lightroom", "gimp", "affinity photo", "paint.net",
    "pixelmator", "canva", "snapseed", "picsart", "photopea",
)


def compute(meta_result, ela_result, fft_result, noise_result, jpeg_result,
            face_result, watermark_result=None, ml_result=None) -> dict:

    if watermark_result is None:
        watermark_result = {"score": 0.0, "detected": False, "flags": []}
    if ml_result is None:
        ml_result = {"score": 0.0, "ai_prob": 0.0, "deepfake_prob": 0.0, "real_prob": 1.0,
                     "flags": [], "available": False}

    ml_available, deepfake_prob = ml_result.get("available", False), float(ml_result.get("deepfake_prob", 0.0) or 0.0)
    wm_detected = watermark_result.get("detected", False)
    software = (meta_result.get("software") or "").strip()
    software_lower = software.lower()
    editing_software = next((name for name in EDITING_SOFTWARE if name in software_lower), None)
    ela_std = float(ela_result.get("std_diff", 0.0) or 0.0)
    edit_reasons = []
    if editing_software:
        edit_reasons.append(f"editing software metadata: {software}")
    if ela_std > 15.0:
        edit_reasons.append(f"high ELA variance (std={ela_std:.1f})")
    possible_watermark_edit = bool(wm_detected and edit_reasons)
    watermark_edit_reason = "; ".join(edit_reasons) if possible_watermark_edit else None

    # ── Build score map ───────────────────────────────────────────────────
    heuristic_scores = {
        "metadata": meta_result.get("score", 0.0),
        "ela":      ela_result.get("score",  0.0),
        "fft":      fft_result.get("score",  0.0),
        "noise":    noise_result.get("score", 0.0),
        "jpeg":     jpeg_result.get("score",  0.0),
        "face":     face_result.get("score",  0.0) if face_result.get("face_detected") else 0.0,
    }

    if ml_available:
        weights = dict(WEIGHTS_ML)
        scores  = {"ml": ml_result["ai_prob"], **heuristic_scores}
    else:
        weights = dict(WEIGHTS_HEURISTIC)
        scores  = dict(heuristic_scores)

    # ── Context corrections ───────────────────────────────────────────────
    # No face → redistribute face weight
    if not face_result.get("face_detected") and "face" in weights:
        extra = weights.pop("face", 0.0)
        scores.pop("face", None)
        for k in ("noise", "ela", "metadata"):
            if k in weights:
                weights[k] += extra / 3.0

    # Face looks natural → slightly reduce noise and fft
    if face_result.get("face_detected") and scores.get("face", 1.0) < 0.25:
        scores["noise"] = max(0.0, scores.get("noise", 0.0) - 0.12)
        scores["fft"]   = max(0.0, scores.get("fft",   0.0) - 0.08)

    # ── Weighted sum ──────────────────────────────────────────────────────
    total_w = sum(weights.get(k, 0.0) for k in scores)
    ai_prob = sum(scores[k] * weights.get(k, 0.0) for k in scores) / (total_w + 1e-9)
    ai_prob = float(max(0.0, min(1.0, ai_prob)))

    if wm_detected:
        wm_weight = EDITED_WATERMARK_WEIGHT if possible_watermark_edit else WATERMARK_WEIGHT
        wm_score = float(watermark_result.get("score", 0.0))
        weights["watermark"] = wm_weight
        # A visual mark can be copied, so it must not overwhelm image evidence.
        ai_prob = min(1.0, ai_prob + wm_score * wm_weight)

    real_prob = 1.0 - ai_prob
    ai_pct    = round(ai_prob * 100)
    real_pct  = 100 - ai_pct

    # ── Verdict ───────────────────────────────────────────────────────────
    if ai_prob >= 0.60 or (ml_available and deepfake_prob >= 0.60 and deepfake_prob >= ml_result.get("ai_prob", 0.0)):
        verdict = "Likely Deepfake / Manipulated" if ml_available and deepfake_prob >= 0.60 and deepfake_prob >= ml_result.get("ai_prob", 0.0) else "Likely AI Generated"
    elif ai_prob <= 0.35:
        verdict = "Likely Real Photo"
    else:
        verdict = "Uncertain / Inconclusive"

    # ── Confidence ────────────────────────────────────────────────────────
    margin = abs(ai_prob - 0.5)
    if ml_available and (margin > 0.35 or deepfake_prob >= 0.85):
        confidence = "High"
    elif margin > 0.18 or (ml_available and deepfake_prob >= 0.60):
        confidence = "Medium"
    else:
        confidence = "Low"

    # ── Evidence ──────────────────────────────────────────────────────────
    evidence = []
    # ML result first (most important)
    if ml_available:
        evidence.extend(ml_result.get("flags", []))
    if wm_detected:
        evidence.extend(watermark_result.get("flags", []))
        evidence.append("Watermark origin is unverified; a visual match alone cannot prove who added it.")
    if possible_watermark_edit:
        evidence.append(
            f"Possible post-processing signal ({watermark_edit_reason}); this does not prove the watermark was added later."
        )
    for r in [meta_result, ela_result, fft_result, noise_result, jpeg_result, face_result]:
        evidence.extend(r.get("flags", []))
    if not ml_available:
        evidence.extend(ml_result.get("flags", []))
    evidence.extend(watermark_result.get("flags", []) if not wm_detected else [])

    # Build per-module score record for UI
    all_scores = {**scores}
    if wm_detected:
        all_scores["watermark"] = watermark_result.get("score", 0.0)

    module_scores = {
        k: {
            "raw":    round(all_scores[k], 3),
            "weight": round(weights.get(k, 0.0), 3),
            "contrib":round(all_scores[k] * weights.get(k, 0.0), 3),
        }
        for k in all_scores
    }

    return dict(
        ai_probability   = round(ai_prob, 4),
        real_probability = round(real_prob, 4),
        ai_pct      = ai_pct,
        real_pct    = real_pct,
        verdict     = verdict,
        confidence  = confidence,
        evidence    = evidence,
        module_scores = module_scores,
        watermark_detected = wm_detected,
        watermark_origin_verified = False,
        possible_watermark_edit = possible_watermark_edit,
        watermark_editing_software = software if possible_watermark_edit else None,
        watermark_edit_reason = watermark_edit_reason,
        ml_available = ml_available, deepfake_probability = round(deepfake_prob, 4),
    )
