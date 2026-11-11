"""
modules/face.py
Face analysis using MediaPipe FaceLandmarker (Tasks API — mediapipe >= 0.10).

Falls back gracefully if the model file is not available or the API changes.

Checks:
  - Eye symmetry (inter-pupil distance ratio)
  - Ear landmark Y-alignment
  - Facial proportion consistency
  - Overall deepfake probability estimate

Only runs if a face is detected.
"""

import os
import urllib.request
import numpy as np
from PIL import Image

# ── Model file (downloaded once, cached locally) ────────────────────────────
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_DIR  = os.path.join(os.path.dirname(__file__), "_models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")


def _ensure_model() -> str | None:
    """Download the FaceLandmarker model if not already cached. Returns path or None."""
    os.makedirs(_MODEL_DIR, exist_ok=True)
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        return _MODEL_PATH
    except Exception as e:
        return None


def _euclidean_2d(ax, ay, bx, by) -> float:
    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


def _lm_xy(lm_list, idx):
    """Return (x, y) for a landmark by index."""
    lm = lm_list[idx]
    return lm.x, lm.y


def _analyze_landmarks(face_landmarks) -> dict:
    """
    Compute facial geometry metrics from MediaPipe FaceLandmarker output.
    Landmark indices are the same 478-point mesh as Face Mesh.
    """
    lm = face_landmarks

    LEFT_EYE_INNER  = 133
    RIGHT_EYE_INNER = 362
    LEFT_EYE_OUTER  = 33
    RIGHT_EYE_OUTER = 263
    LEFT_EAR        = 234
    RIGHT_EAR       = 454
    CHIN            = 152
    FOREHEAD        = 10

    def d(a, b):
        ax, ay = _lm_xy(lm, a)
        bx, by = _lm_xy(lm, b)
        return _euclidean_2d(ax, ay, bx, by)

    left_eye_w  = d(LEFT_EYE_OUTER,  LEFT_EYE_INNER)
    right_eye_w = d(RIGHT_EYE_OUTER, RIGHT_EYE_INNER)
    eye_sym = 1.0 - abs(left_eye_w - right_eye_w) / (max(left_eye_w, right_eye_w) + 1e-6)

    _, left_ear_y  = _lm_xy(lm, LEFT_EAR)
    _, right_ear_y = _lm_xy(lm, RIGHT_EAR)
    ear_align = 1.0 - min(abs(left_ear_y - right_ear_y) * 10, 1.0)

    face_h = d(FOREHEAD, CHIN)
    face_w = d(LEFT_EAR, RIGHT_EAR)
    proportion = face_h / (face_w + 1e-6)
    prop_score = 1.0 - float(np.clip(abs(proportion - 1.5) / 1.0, 0.0, 1.0))

    return {
        "eye_symmetry":    round(eye_sym, 3),
        "ear_alignment":   round(ear_align, 3),
        "face_proportion": round(proportion, 3),
        "proportion_score": round(prop_score, 3),
    }


def _draw_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    """Manually draw landmarks as dots on the image (no solutions.drawing_utils needed)."""
    annotated = rgb_image.copy()
    h, w = annotated.shape[:2]

    import cv2

    for face_landmarks in detection_result.face_landmarks:
        for lm in face_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 1, (0, 200, 255), -1)

    return annotated


# ── Public API ───────────────────────────────────────────────────────────────

def analyze(pil_image: Image.Image) -> dict:
    """
    Run face detection and landmark analysis on pil_image.

    Returns dict with keys:
        face_detected, num_faces, metrics, deepfake_prob,
        score (0-1), flags, annotated_image
    """
    result = {
        "face_detected":  False,
        "num_faces":      0,
        "metrics":        None,
        "deepfake_prob":  None,
        "score":          0.0,
        "flags":          [],
        "annotated_image": None,
    }

    # ── Import MediaPipe Tasks API ───────────────────────────────────────
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except Exception as e:
        result["flags"].append(f"MediaPipe import failed — face analysis skipped ({e})")
        return result

    # ── Ensure model is available ────────────────────────────────────────
    model_path = _ensure_model()
    if model_path is None:
        result["flags"].append(
            "FaceLandmarker model unavailable (network error) — face analysis skipped"
        )
        return result

    # ── Run inference ────────────────────────────────────────────────────
    rgb_image = np.array(pil_image.convert("RGB"))

    try:
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=4,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
        )

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        with mp_vision.FaceLandmarker.create_from_options(options) as detector:
            detection_result = detector.detect(mp_img)

    except Exception as e:
        result["flags"].append(f"FaceLandmarker inference error — face analysis skipped ({e})")
        return result

    if not detection_result.face_landmarks:
        result["flags"].append("No face detected — face analysis skipped")
        return result

    result["face_detected"] = True
    result["num_faces"] = len(detection_result.face_landmarks)

    # ── Annotated image ──────────────────────────────────────────────────
    annotated = _draw_landmarks_on_image(rgb_image, detection_result)
    result["annotated_image"] = Image.fromarray(annotated)

    # ── Geometry metrics (first face) ────────────────────────────────────
    metrics = _analyze_landmarks(detection_result.face_landmarks[0])
    result["metrics"] = metrics

    # ── Scoring ──────────────────────────────────────────────────────────
    score = 0.0
    flags = []

    if metrics["eye_symmetry"] < 0.80:
        score += 0.30
        flags.append(f"Eye asymmetry detected (symmetry={metrics['eye_symmetry']:.2f})")
    elif metrics["eye_symmetry"] > 0.98:
        score += 0.15
        flags.append(f"Near-perfect eye symmetry ({metrics['eye_symmetry']:.2f}) — unusually symmetric")
    else:
        flags.append(f"Eye symmetry normal ({metrics['eye_symmetry']:.2f})")

    if metrics["ear_alignment"] < 0.70:
        score += 0.25
        flags.append(f"Ear misalignment detected (score={metrics['ear_alignment']:.2f})")
    else:
        flags.append(f"Ear alignment normal ({metrics['ear_alignment']:.2f})")

    if metrics["proportion_score"] < 0.50:
        score += 0.20
        flags.append(f"Unusual face proportions (ratio={metrics['face_proportion']:.2f})")
    else:
        flags.append(f"Face proportions plausible (ratio={metrics['face_proportion']:.2f})")

    if result["num_faces"] > 2:
        score += 0.10
        flags.append(f"Multiple faces detected ({result['num_faces']}) — check for artefacts")

    score = float(np.clip(score, 0.0, 1.0))
    result["deepfake_prob"] = round(score, 3)
    result["score"]         = round(score, 3)
    result["flags"]         = flags

    return result
