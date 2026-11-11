"""Optional AI-image ensemble for slower, higher-signal analysis."""
import os

# Keep model downloads outside the watched source tree, especially on Linux.
_CACHE = os.environ.get("HF_HOME") or (
    r"C:\hfc" if os.name == "nt"
    else os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
)
os.makedirs(_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", _CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_CACHE, "hub"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_CACHE, "hub"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from PIL import Image

MODEL_ID_1 = "Ateeqq/ai-vs-human-image-detector"
MODEL_ID_2 = "dima806/deepfake_vs_real_image_detection"

def _get_pipelines(model_id_1: str = MODEL_ID_1, model_id_2: str = MODEL_ID_2):
    """Load and return an ensemble of HuggingFace pipelines. Heavy – cache with st.cache_resource."""
    from transformers import pipeline as hf_pipeline
    import transformers.utils.logging as hf_logging
    hf_logging.disable_progress_bar()
    
    pipe1 = hf_pipeline("image-classification", model=model_id_1)
    pipe2 = hf_pipeline("image-classification", model=model_id_2)
    return (pipe1, pipe2)


def analyze(pil_image: Image.Image, pipes=None) -> dict:
    """
    Run the pretrained AI image classifier ensemble.
    Pass `pipes` from st.cache_resource to avoid reloading the models.
    """
    flags = []

    if pipes is None or len(pipes) != 2:
        return {
            "score": 0.0, "ai_prob": 0.0, "real_prob": 1.0,
            "flags": ["⚠ ML classifiers not loaded — results are heuristic only"],
            "available": False, "label": "unavailable",
        }

    try:
        img_rgb = pil_image.convert("RGB").resize((224, 224))
        
        # Run inference on both models
        res1 = pipes[0](img_rgb)
        res2 = pipes[1](img_rgb)

        def _get_ai_prob(results, ai_keywords):
            a_p = 0.0
            r_p = 0.0
            for r in results:
                lbl = r["label"].lower()
                if any(k in lbl for k in ai_keywords):
                    a_p += float(r["score"])
                else:
                    r_p += float(r["score"])
            
            total = a_p + r_p
            return a_p / total if total > 0 else 0.0

        ai_kws_1 = ("deepfake", "fake", "ai", "artificial")
        ai_kws_2 = ("fake", "ai", "synthetic")

        prob1 = _get_ai_prob(res1, ai_kws_1)
        prob2 = _get_ai_prob(res2, ai_kws_2)

        # Average the ensemble probabilities
        ai_prob = (prob1 + prob2) / 2.0
        real_prob = 1.0 - ai_prob

        if ai_prob >= 0.85:
            label = "AI Generated"
            flags.append(f"🔴 ML ensemble is highly confident this is AI-generated ({ai_prob:.1%})")
        elif ai_prob >= 0.65:
            label = "Likely AI"
            flags.append(f"🟠 ML ensemble suspects AI generation ({ai_prob:.1%})")
        elif ai_prob >= 0.45:
            label = "Uncertain"
            flags.append(f"🟡 ML ensemble is uncertain (AI: {ai_prob:.1%}, Real: {real_prob:.1%})")
        elif real_prob >= 0.65:
            label = "Likely Real"
            flags.append(f"🟢 ML ensemble is confident this is a real photograph ({real_prob:.1%})")
        else:
            label = "Uncertain"
            flags.append(f"ML ensemble uncertain (AI: {ai_prob:.1%}, Real: {real_prob:.1%})")

        return {
            "score":     round(ai_prob, 4),
            "ai_prob":   round(ai_prob, 4),
            "real_prob": round(real_prob, 4),
            "flags":     flags,
            "available": True,
            "label":     label,
        }

    except Exception as exc:
        return {
            "score": 0.0, "ai_prob": 0.0, "real_prob": 1.0,
            "flags": [f"ML classifier error: {str(exc)[:120]}"],
            "available": False, "label": "error",
        }
