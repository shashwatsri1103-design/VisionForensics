"""Classify images as AI-generated, deepfake/manipulated, or real."""
import os

_CACHE = (
    os.environ.get("VISIONFORENSICS_CACHE")
    or os.environ.get("HF_HOME")
    or (r"C:\hfc" if os.name == "nt" else os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
)
os.makedirs(_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", _CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_CACHE, "hub"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_CACHE, "hub"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from PIL import Image

MODEL_ID = "prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0"


def _get_pipeline():
    """Load the image classifier. Heavy — cache with st.cache_resource."""
    from transformers import pipeline as hf_pipeline
    import transformers.utils.logging as hf_logging
    hf_logging.disable_progress_bar()
    return hf_pipeline("image-classification", model=MODEL_ID)


def _classify_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", " ").replace("-", " ")
    if "deepfake" in normalized or "manipulated" in normalized:
        return "deepfake"
    if any(term in normalized for term in ("artificial", "synthetic", "generated")):
        return "artificial"
    if normalized == "ai":
        return "artificial"
    if "real" in normalized or "authentic" in normalized:
        return "real"
    raise ValueError(f"Unrecognized classifier label: {label}")


def _class_probabilities(results: list[dict]) -> dict[str, float]:
    probabilities = {"artificial": 0.0, "deepfake": 0.0, "real": 0.0}
    found = set()
    for result in results:
        category = _classify_label(result["label"])
        probability = float(result["score"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid classifier probability for {result['label']}")
        probabilities[category] += probability
        found.add(category)

    missing = set(probabilities) - found
    if missing:
        raise ValueError(f"Classifier output is missing classes: {', '.join(sorted(missing))}")
    total = sum(probabilities.values())
    if total <= 0.0:
        raise ValueError("Classifier returned zero probability for every class")
    return {key: value / total for key, value in probabilities.items()}


def analyze(pil_image: Image.Image, pipe=None) -> dict:
    """Run the pretrained three-class classifier using a cached pipeline."""
    if pipe is None:
        return {
            "score": 0.0, "ai_prob": 0.0, "deepfake_prob": 0.0, "real_prob": 1.0,
            "flags": ["⚠ ML classifier not loaded — results are heuristic only"],
            "available": False, "label": "unavailable",
        }

    try:
        probabilities = _class_probabilities(pipe(pil_image.convert("RGB"), top_k=None))
        ai_prob = probabilities["artificial"]
        deepfake_prob = probabilities["deepfake"]
        real_prob = probabilities["real"]
        flags = [
            "ML class probabilities — "
            f"AI-generated: {ai_prob:.1%}; deepfake/manipulated: {deepfake_prob:.1%}; real: {real_prob:.1%}"
        ]

        if deepfake_prob >= 0.60 and deepfake_prob >= ai_prob:
            label = "Likely Deepfake / Manipulated"
            flags.append(f"🟠 ML classifier detects deepfake-style manipulation ({deepfake_prob:.1%})")
        elif ai_prob >= 0.85:
            label = "AI Generated"
            flags.append(f"🔴 ML classifier is highly confident this is AI-generated ({ai_prob:.1%})")
        elif ai_prob >= 0.65:
            label = "Likely AI"
            flags.append(f"🟠 ML classifier suspects AI generation ({ai_prob:.1%})")
        elif ai_prob >= 0.45:
            label = "Uncertain"
            flags.append(f"🟡 ML classifier is uncertain (AI-generated: {ai_prob:.1%})")
        elif real_prob >= 0.65:
            label = "Likely Real"
            flags.append(f"🟢 ML classifier favors a real photograph ({real_prob:.1%})")
        else:
            label = "Uncertain"
            flags.append(f"ML classifier is inconclusive (AI-generated: {ai_prob:.1%}, real: {real_prob:.1%})")

        return {
            "score": round(ai_prob, 4),
            "ai_prob": round(ai_prob, 4),
            "deepfake_prob": round(deepfake_prob, 4),
            "real_prob": round(real_prob, 4),
            "flags": flags,
            "available": True,
            "label": label,
        }
    except Exception as exc:
        return {
            "score": 0.0, "ai_prob": 0.0, "deepfake_prob": 0.0, "real_prob": 1.0,
            "flags": [f"ML classifier error: {str(exc)[:120]}"],
            "available": False, "label": "error",
        }
