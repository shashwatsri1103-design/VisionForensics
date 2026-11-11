import traceback
import os

_CACHE = r"C:\hfc"
os.makedirs(_CACHE, exist_ok=True)
os.environ["HF_HOME"] = _CACHE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_CACHE, "hub")

from transformers import pipeline

try:
    print("Loading pipeline...")
    pipe = pipeline("image-classification", model="umm-maybe/AI-image-detector", cache_dir=os.path.join(_CACHE, "hub"))
    print("Success")
except Exception as e:
    print("Error:")
    traceback.print_exc()
