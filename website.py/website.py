import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import requests
import io
import numpy as np
from PIL import Image, ImageChops, ExifTags
import plotly.graph_objects as go

st.set_page_config(page_title="VisionForensics AI", page_icon="🔍", layout="centered")
st.title("🔍 VisionForensics AI")
st.write("Upload an image to analyze forensic modules and detect AI generation.")
st.caption("Uploaded images are sent to Sightengine for AI analysis.")

def get_sightengine_credentials():
    try:
        return (
            st.secrets["SIGHTENGINE_API_USER"],
            st.secrets["SIGHTENGINE_API_SECRET"],
        )
    except (StreamlitSecretNotFoundError, KeyError):
        st.error(
            "Sightengine credentials are not configured. Add "
            "SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET to Streamlit secrets."
        )
        st.stop()

# --- Lightweight Forensic Calculations for the Radar Chart ---
def get_forensic_breakdown(img, ml_score):
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32)

    # 1. ELA (Error Level Analysis)
    buf = io.BytesIO()
    img_rgb.save(buf, "JPEG", quality=90)
    buf.seek(0)
    resaved = Image.open(buf)
    diff = np.array(ImageChops.difference(img_rgb, resaved), dtype=np.float32)
    ela_score = min(100.0, float(np.mean(diff)) * 8.0)

    # 2. Noise Variance
    gray = np.mean(arr, axis=2)
    noise_residual = np.abs(gray - np.roll(gray, 1, axis=0))
    noise_score = min(100.0, float(np.std(noise_residual)) * 3.5)

    # 3. FFT (Frequency Domain)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    fft_score = min(100.0, max(0.0, (float(np.mean(magnitude)) - 100) * 1.5))

    # 4. Metadata Check (Missing EXIF increases suspicion score)
    exif = img.getexif()
    metadata_score = 20.0 if (exif and len(exif) > 3) else 75.0

    # 5. JPEG Quantization / Compression check
    jpeg_score = 65.0 if img.format != "JPEG" else 35.0

    return {
        "ML": round(ml_score, 1),
        "METADATA": round(metadata_score, 1),
        "ELA": round(ela_score, 1),
        "FFT": round(fft_score, 1),
        "NOISE": round(noise_score, 1),
        "JPEG": round(jpeg_score, 1),
    }

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch", alt="Uploaded image preview")

    with st.spinner("Running forensic modules and AI analysis..."):
        api_user, api_secret = get_sightengine_credentials()
        try:
            response = requests.post(
                "https://api.sightengine.com/1.0/check.json",
                files={"media": uploaded_file.getvalue()},
                data={
                    "models": "genai",
                    "api_user": api_user,
                    "api_secret": api_secret,
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException:
            st.error("Sightengine could not analyze the image. Check the API secrets and service availability.")
            st.stop()

    if result.get("status") == "success":
        ai_prob = result["type"]["ai_generated"]
        percentage = round(ai_prob * 100, 1)

        st.subheader("Detection Verdict")
        st.progress(ai_prob)

        if percentage > 50:
            st.error(f"🚨 **AI-Generated** ({percentage}% probability)")
        else:
            st.success(f"✅ **Likely Real** ({100 - percentage:.1f}% real / {percentage}% AI)")

        # --- Build the Radar Chart ("MODULE BREAKDOWN") ---
        scores = get_forensic_breakdown(image, percentage)
        categories = ["ML", "METADATA", "ELA", "FFT", "NOISE", "JPEG"]
        values = [scores[cat] for cat in categories]

        # Close the polygon loop
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]

        st.markdown("### 📊 MODULE BREAKDOWN")

        fig = go.Figure(
            data=[
                go.Scatterpolar(
                    r=values_closed,
                    theta=categories_closed,
                    fill="toself",
                    fillcolor="rgba(255, 107, 107, 0.25)",
                    line=dict(color="#FF6B6B", width=3),
                    marker=dict(color="#FF6B6B", size=8),
                )
            ]
        )

        fig.update_layout(
            polar=dict(
                bgcolor="#0E1117",
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showticklabels=False,
                    gridcolor="#262730",
                    linecolor="#262730",
                ),
                angularaxis=dict(
                    tickfont=dict(size=13, color="#8B949E"),
                    gridcolor="#262730",
                    linecolor="#262730",
                ),
            ),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            showlegend=False,
            margin=dict(l=50, r=50, t=40, b=40),
            height=450,
        )

        st.plotly_chart(
            fig,
            width="stretch",
            alt="Forensic module suspicion score breakdown",
        )

        # Optional: Show exact numbers below the chart
        cols = st.columns(6)
        for idx, cat in enumerate(categories):
            cols[idx].metric(cat, f"{scores[cat]}%")
    else:
        st.error("API Error: Check your API keys or internet connection.")
