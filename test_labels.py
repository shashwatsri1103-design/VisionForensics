import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from transformers import AutoConfig

m1 = "prithivMLmods/Deep-Fake-Detector-v2-Model"
# m2 is incompatible with standard transformers pipeline due to custom architecture "ai-image-detector"
m3 = "dima806/deepfake_vs_real_image_detection"

c1 = AutoConfig.from_pretrained(m1)
c3 = AutoConfig.from_pretrained(m3)

print("M1 classes:", c1.id2label)
print("M3 classes:", c3.id2label)
