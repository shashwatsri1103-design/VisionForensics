"""
modules/metadata.py  — v2
"""
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from typing import Any


def _decode_gps(gps_info: dict) -> str | None:
    try:
        def to_decimal(v):
            d, m, s = v
            return d + m/60.0 + s/3600.0
        lat = to_decimal(gps_info.get(2,(0,0,0)))
        lon = to_decimal(gps_info.get(4,(0,0,0)))
        if gps_info.get(1,"N") == "S": lat = -lat
        if gps_info.get(3,"E") == "W": lon = -lon
        return f"{lat:.6f}°, {lon:.6f}°"
    except Exception:
        return None


def _safe(val: Any) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore").strip("\x00")
    if isinstance(val, tuple) and len(val) == 2:
        try:
            return str(val[0]/val[1]) if val[1] != 0 else str(val[0])
        except Exception:
            return str(val)
    return str(val)


def analyze(pil_image: Image.Image, file_size_bytes: int) -> dict:
    result = dict(found=False, camera_model=None, lens=None, date_taken=None,
                  software=None, gps=None, raw_tags={}, score=0.0, flags=[])

    try:
        exif_data = pil_image._getexif()  # type: ignore
    except Exception:
        exif_data = None

    if not exif_data:
        # ── Missing EXIF ──────────────────────────────────────────────────
        # Real score: 0.35 — many legit photos lose EXIF via WhatsApp/social.
        # Previously was 0.70 which caused large false-positive contribution.
        result["score"] = 0.35
        result["flags"].append("No EXIF metadata present")
        return result

    result["found"] = True
    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, str(tag_id))
        if tag == "GPSInfo":
            result["gps"] = _decode_gps(value)
            continue
        readable = _safe(value)
        result["raw_tags"][tag] = readable
        tl = tag.lower()
        if "model"    in tl and not result["camera_model"]: result["camera_model"] = readable
        if "lens"     in tl and not result["lens"]:         result["lens"]         = readable
        if "datetime" in tl and not result["date_taken"]:   result["date_taken"]   = readable
        if "software" in tl and not result["software"]:     result["software"]     = readable

    suspicion = 0.0
    if not result["camera_model"]:
        suspicion += 0.20   # reduced from 0.25
        result["flags"].append("No camera model in EXIF")
    if not result["date_taken"]:
        suspicion += 0.08
        result["flags"].append("No capture date in EXIF")

    sw = (result["software"] or "").lower()
    ai_kw = ["stable diffusion","midjourney","dall","firefly","generative",
              "dream","invoke","comfy","imagen","gemini"]
    if any(k in sw for k in ai_kw):
        suspicion += 0.65
        result["flags"].append(f"Software field suggests AI tool: {result['software']}")

    result["score"] = min(suspicion, 1.0)
    if result["score"] < 0.15:
        result["flags"].append("Rich EXIF metadata — consistent with real camera")
    return result
