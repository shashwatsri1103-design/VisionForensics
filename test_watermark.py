import numpy as np
from PIL import Image, ImageDraw

from modules.report import compute
from modules.watermark import detect_sparkle


def _textured_image(with_watermark: bool) -> Image.Image:
    width, height = 640, 360
    rng = np.random.default_rng(42)
    base = np.tile(np.linspace(90, 150, width, dtype=np.float32), (height, 1))
    texture = rng.normal(0, 7, (height, width)).astype(np.float32)
    gray = np.clip(base + texture, 0, 255).astype(np.uint8)
    image = Image.fromarray(np.dstack((gray, gray, gray)))

    if with_watermark:
        radius = 15
        center_x, center_y = width - 52, height - 48
        points = []
        for angle in np.linspace(0, 2 * np.pi, 65):
            x = center_x + radius * np.cos(angle) ** 3
            y = center_y + radius * np.sin(angle) ** 3
            points.append((x, y))

        overlay = Image.new("RGBA", image.size)
        ImageDraw.Draw(overlay).polygon(points, fill=(255, 255, 255, 150))
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    return image


def test_detects_translucent_corner_sparkle():
    result = detect_sparkle(_textured_image(with_watermark=True))

    assert result["detected"]
    assert result["location"] == "bottom-right"


def test_rejects_textured_image_without_sparkle():
    result = detect_sparkle(_textured_image(with_watermark=False))

    assert not result["detected"]


def test_visual_watermark_is_not_treated_as_authenticated_provenance():
    empty = {"score": 0.0, "flags": []}
    face = {"score": 0.0, "face_detected": False, "flags": []}
    ml = {"score": 0.47, "ai_prob": 0.47, "real_prob": 0.53, "available": True, "flags": []}
    mark = detect_sparkle(_textured_image(with_watermark=True))
    baseline = compute(empty, empty, empty, empty, empty, face, None, ml)
    result = compute(
        empty,
        empty,
        empty,
        empty,
        empty,
        face,
        mark,
        ml,
    )

    assert result["watermark_detected"]
    assert not result["watermark_origin_verified"]
    assert result["ai_pct"] >= baseline["ai_pct"] + 25
    assert result["ai_pct"] >= 85
    assert result["module_scores"]["watermark"]["weight"] == 0.6


def test_editor_software_metadata_reduces_watermark_contribution():
    metadata = {"score": 0.0, "flags": [], "software": "Adobe Photoshop"}
    empty = {"score": 0.0, "flags": []}
    face = {"score": 0.0, "face_detected": False, "flags": []}
    ml = {"score": 0.47, "ai_prob": 0.47, "real_prob": 0.53, "available": True, "flags": []}
    mark = detect_sparkle(_textured_image(with_watermark=True))
    baseline = compute(empty, empty, empty, empty, empty, face, None, ml)
    result = compute(
        metadata,
        empty,
        empty,
        empty,
        empty,
        face,
        mark,
        ml,
    )

    assert result["possible_watermark_edit"]
    assert result["watermark_editing_software"] == "Adobe Photoshop"
    assert result["module_scores"]["watermark"]["weight"] == 0.08
    assert result["ai_pct"] < baseline["ai_pct"] + 10


def test_high_ela_variance_reduces_watermark_contribution():
    empty = {"score": 0.0, "flags": []}
    face = {"score": 0.0, "face_detected": False, "flags": []}
    ml = {"score": 0.47, "ai_prob": 0.47, "real_prob": 0.53, "available": True, "flags": []}
    ela = {"score": 0.0, "std_diff": 18.0, "flags": []}
    result = compute(
        empty,
        ela,
        empty,
        empty,
        empty,
        face,
        detect_sparkle(_textured_image(with_watermark=True)),
        ml,
    )

    assert result["possible_watermark_edit"]
    assert "high ELA variance" in result["watermark_edit_reason"]
    assert result["module_scores"]["watermark"]["weight"] == 0.08