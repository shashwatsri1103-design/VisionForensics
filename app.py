"""
app.py – VisionForensics AI v2
Run with: streamlit run app.py
"""
import io, math, os, sys
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from modules import metadata as meta_mod, fft as fft_mod, noise as noise_mod
from modules import jpeg as jpeg_mod, face as face_mod, report as report_mod
from modules import watermark as wm_mod, ela as ela_mod
import modules.ai_classifier as ac_mod

@st.cache_resource(show_spinner="⏳ Loading AI Detection Model (one-time ~1-3 mins)...")
def load_ml_pipe(model_id_1: str, model_id_2: str):
    """Cache pipelines by model IDs so changing detectors invalidates the cache."""
    # Do NOT wrap this in a try/except that returns None!
    # If we return None on a network timeout, Streamlit caches 'None' permanently
    # for the rest of the session. By letting the exception raise, the user can 
    # just refresh the page to try the download again.
    return ac_mod._get_pipelines(model_id_1, model_id_2)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="VisionForensics AI", page_icon="🔬",
                   layout="wide", initial_sidebar_state="collapsed")

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#080c16 0%,#0c1120 60%,#080d18 100%);color:#e2e8f0;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.2rem;padding-bottom:3rem;}

/* Hero */
.vf-hero{background:linear-gradient(135deg,#111827,#0f172a);border:1px solid rgba(99,179,237,.12);
  border-radius:20px;padding:2.2rem 2.8rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}
.vf-hero::after{content:"";position:absolute;top:-80px;right:-80px;width:260px;height:260px;
  background:radial-gradient(circle,rgba(99,179,237,.09) 0%,transparent 70%);border-radius:50%;}
.vf-hero h1{font-size:2.4rem;font-weight:800;
  background:linear-gradient(135deg,#63b3ed,#a78bfa,#f472b6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 .35rem;}
.vf-hero p{color:#64748b;font-size:.95rem;margin:0;}

/* Section heading */
.vf-heading{font-size:.8rem;font-weight:700;color:#63b3ed;letter-spacing:.1em;
  text-transform:uppercase;margin-bottom:.8rem;display:flex;align-items:center;gap:.4rem;}

/* Evidence */
.ev-item{display:flex;align-items:flex-start;gap:.7rem;padding:.55rem .9rem;
  border-radius:10px;margin:.35rem 0;font-size:.86rem;}
.ev-high{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);}
.ev-med{background:rgba(251,146,60,.07);border:1px solid rgba(251,146,60,.18);}
.ev-ok{background:rgba(52,211,153,.07);border:1px solid rgba(52,211,153,.18);}
.ev-icon{font-size:1rem;flex-shrink:0;margin-top:.05rem;}
.ev-text-high{color:#fca5a5;}.ev-text-med{color:#fdba74;}.ev-text-ok{color:#6ee7b7;}

/* Module score pill */
.mod-pill{text-align:center;padding:.9rem .5rem;border-radius:14px;}
.mod-pill .num{font-family:'JetBrains Mono',monospace;font-size:1.8rem;font-weight:800;line-height:1;}
.mod-pill .lbl{font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:#64748b;margin-top:.25rem;}
.mod-pill .wt{font-size:.65rem;color:#374151;margin-top:.1rem;}

/* Stats strip */
.stat-chip{display:inline-flex;align-items:center;gap:.4rem;background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.07);border-radius:8px;padding:.3rem .75rem;
  font-size:.8rem;color:#94a3b8;font-family:'JetBrains Mono',monospace;}

/* Progress bar */
.pb-wrap{background:rgba(255,255,255,.06);border-radius:100px;height:7px;margin:.4rem 0 .2rem;}
.pb-fill{height:7px;border-radius:100px;}
.pb-label{font-size:.72rem;color:#64748b;}

/* Verdict badge */
.verdict{display:inline-block;padding:.55rem 1.8rem;border-radius:100px;
  font-size:1rem;font-weight:700;letter-spacing:.03em;}
.v-ai{background:rgba(239,68,68,.12);color:#fca5a5;border:1.5px solid rgba(239,68,68,.4);}
.v-real{background:rgba(52,211,153,.1);color:#6ee7b7;border:1.5px solid rgba(52,211,153,.35);}
.v-unk{background:rgba(251,146,60,.1);color:#fdba74;border:1.5px solid rgba(251,146,60,.35);}

/* Confidence */
.conf-badge{display:inline-block;padding:.25rem .9rem;border-radius:6px;
  font-size:.75rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-left:.6rem;}
.c-high{background:rgba(52,211,153,.15);color:#6ee7b7;}
.c-med{background:rgba(251,146,60,.15);color:#fdba74;}
.c-low{background:rgba(239,68,68,.15);color:#fca5a5;}

/* Tab override */
button[data-baseweb="tab"]{font-size:.82rem!important;font-weight:600!important;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────
def heading(icon, txt):
    st.markdown(f'<div class="vf-heading">{icon} {txt}</div>', unsafe_allow_html=True)

def pbar(score, color="#ef4444"):
    pct = int(score * 100)
    st.markdown(f"""
    <div class="pb-wrap"><div class="pb-fill" style="width:{pct}%;background:{color};"></div></div>
    <div class="pb-label">{pct}% suspicion</div>""", unsafe_allow_html=True)

def mod_pill(label, score, weight):
    color = "#ef4444" if score > .55 else "#fb923c" if score > .28 else "#34d399"
    st.markdown(f"""
    <div class="mod-pill" style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);">
      <div class="num" style="color:{color};">{int(score*100)}%</div>
      <div class="lbl">{label}</div>
      <div class="wt">weight {weight:.0%}</div>
    </div>""", unsafe_allow_html=True)

def gauge_svg(pct, color):
    pct = min(pct, 99)
    angle = math.pi * pct / 100
    r, cx, cy = 78, 100, 98
    ex = cx - r * math.cos(angle)
    ey = cy - r * math.sin(angle)
    # For a semi-circle, the angle is never > 180 degrees, so large-arc-flag (la) MUST always be 0.
    # Setting it to 1 forces the SVG renderer to shift the circle center to find a >180 deg arc.
    track  = f"M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}"
    filled = f"M {cx-r} {cy} A {r} {r} 0 0 1 {ex:.2f} {ey:.2f}"
    return f"""<div style="text-align:center;padding:.5rem 0">
    <svg viewBox="0 0 200 118" width="300" style="overflow:visible">
      <path d="{track}" stroke="rgba(255,255,255,.05)" stroke-width="15" fill="none" stroke-linecap="round"/>
      <path d="{filled}" stroke="{color}" stroke-width="15" fill="none" stroke-linecap="round"
            style="filter:drop-shadow(0 0 10px {color}66)"/>
      <text x="100" y="90" text-anchor="middle" font-family="JetBrains Mono,monospace"
            font-size="42" font-weight="800" fill="{color}">{pct}%</text>
      <text x="100" y="112" text-anchor="middle" font-family="Inter,sans-serif"
            font-size="9" font-weight="700" fill="#374151" letter-spacing="2.5">AI PROBABILITY</text>
    </svg></div>"""

def radar_img(ms):
    # Exclude watermark from radar since it's a binary corner-case
    keys = [k for k in ms.keys() if k != 'watermark']
    labels = [k.upper() for k in keys]
    vals   = [ms[k]['raw'] for k in keys]
    N = len(keys)
    angles = [n/N*2*math.pi for n in range(N)] + [0]
    vals_p = vals + vals[:1]
    fig, ax = plt.subplots(figsize=(3.8,3.8), subplot_kw=dict(polar=True), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.fill(angles, vals_p, alpha=.18, color='#f87171')
    ax.plot(angles, vals_p, 'o-', lw=2, color='#f87171', ms=4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='#94a3b8', size=8.5)
    ax.set_ylim(0,1); ax.set_yticks([.25,.5,.75,1])
    ax.set_yticklabels(['','','','']); ax.grid(color=(1,1,1,0.07), lw=.8)
    ax.spines['polar'].set_color((1,1,1,0.08))
    fig.tight_layout(pad=.2)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig); buf.seek(0)
    return Image.open(buf).copy()

HIGH_KW = ["no exif","no camera","no capture","grid artifact","high-frequency","low noise",
           "smooth","power law","asymmetr","misalign","re-encode","ai tool","png format","webp format",
           "no shot-noise correlation"]
MED_KW  = ["below-average","multiple face","uniform","high file density","high re-encode",
           "inference error","analysis skipped","not loaded","uncertain"]

def ev_class(text):
    t = text.lower()
    if "watermark detected" in t:
        return "high"
    if any(k in t for k in HIGH_KW): return "high"
    if any(k in t for k in MED_KW):  return "med"
    return "ok"

def render_evidence(evidence):
    icon_map = {"high":"🔴","med":"🟠","ok":"🟢"}
    for ev in evidence:
        c = ev_class(ev)
        st.markdown(f"""
        <div class="ev-item ev-{c}">
          <span class="ev-icon">{icon_map[c]}</span>
          <span class="ev-text-{c}">{ev}</span>
        </div>""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="vf-hero">
  <h1>🔬 VisionForensics AI</h1>
  <p>Multi-signal forensic analysis · Metadata · FFT · Noise · Compression · Face Geometry · Explainable Report</p>
</div>""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────
up_col, prev_col = st.columns([1,1], gap="large")
with up_col:
    heading("📤","Upload Image")
    uploaded = st.file_uploader("img", type=["jpg","jpeg","png","webp"],
                                 label_visibility="collapsed")

if uploaded is None:
    st.markdown('<div style="text-align:center;padding:5rem;color:#374151;font-size:.95rem;">⬆ Upload an image above to begin forensic analysis</div>',
                unsafe_allow_html=True)
    st.stop()

# ── Load ──────────────────────────────────────────────────────────────────
raw     = uploaded.read()
fsz     = len(raw)
fmt     = (uploaded.type or "image/jpeg").split("/")[-1].upper()
if fmt == "JPG": fmt = "JPEG"
pil_img = Image.open(io.BytesIO(raw))
pil_rgb = pil_img.convert("RGB")
W, H    = pil_img.size

with prev_col:
    heading("🖼","Preview")
    st.image(pil_rgb, use_container_width=True)
    st.markdown(
        f'<span class="stat-chip">📐 {W}×{H}</span>&nbsp;'
        f'<span class="stat-chip">📄 {fmt}</span>&nbsp;'
        f'<span class="stat-chip">💾 {fsz/1024:.1f} KB</span>',
        unsafe_allow_html=True)

    # Keep the fast forensic signals available without waiting for model downloads.
    use_ml_classifier = st.toggle(
      "Use pretrained ML classifier",
      value=False,
      help="Loads two image classifiers from Hugging Face. The first run downloads them and may take a few minutes.",
    )

# ── Analyse ───────────────────────────────────────────────────────────────
    ml_pipes = load_ml_pipe(ac_mod.MODEL_ID_1, ac_mod.MODEL_ID_2) if use_ml_classifier else None
with st.spinner("🔍 Running forensic modules…"):
    mr = meta_mod.analyze(pil_img, fsz)
    er = ela_mod.analyze(pil_rgb)
    fr = fft_mod.analyze(pil_rgb)
    nr = noise_mod.analyze(pil_rgb)
    jr = jpeg_mod.analyze(pil_rgb, fsz, fmt)
    fc = face_mod.analyze(pil_rgb)
    wr = wm_mod.detect_sparkle(pil_rgb)
    cr = ac_mod.analyze(pil_rgb, pipes=ml_pipes)
    rr = report_mod.compute(mr, er, fr, nr, jr, fc, wr, cr)

ai_pct  = rr["ai_pct"]
verdict = rr["verdict"]
conf    = rr["confidence"]
color   = "#ef4444" if ai_pct>=60 else "#34d399" if ai_pct<=35 else "#fb923c"
vcls    = "v-ai" if "AI" in verdict else "v-real" if "Real" in verdict else "v-unk"
ccls    = {"High":"c-high","Medium":"c-med","Low":"c-low"}.get(conf,"c-low")
ms          = rr["module_scores"]
wm_detected = rr.get("watermark_detected", False)
ml_ok       = rr.get("ml_available", False)

# ── Tabs ──────────────────────────────────────────────────────────────────
st.markdown("---")
t_ov, t_meta, t_ela, t_fft, t_noise, t_jpeg, t_face = st.tabs(
    ["📋 Overview","🗂 Metadata","🎨 ELA","📡 FFT","🌊 Noise","📦 JPEG","👤 Face"])

# ─────────────────────────────────────────────────────────────────────────
# OVERVIEW TAB
# ─────────────────────────────────────────────────────────────────────────
with t_ov:
    # ── ML classifier banner ──────────────────────────────────────
    if ml_ok:
        ml_ai  = int(cr["ai_prob"]*100)
        ml_lbl = cr.get("label", "")
        ml_col = "#ef4444" if ml_ai>=65 else "#34d399" if ml_ai<=35 else "#fb923c"
        st.markdown(f"""
        <div style="background:rgba(99,179,237,.07);border:1px solid rgba(99,179,237,.2);
          border-radius:12px;padding:.75rem 1.1rem;margin-bottom:.7rem;display:flex;
          align-items:center;gap:.9rem;">
          <span style="font-size:1.3rem;">🤖</span>
          <div style="flex:1;">
            <span style="color:#93c5fd;font-weight:700;font-size:.85rem;">
              Pretrained ML Classifier (Dual-Model Ensemble):</span>&nbsp;
            <span style="color:{ml_col};font-weight:800;font-size:.95rem;">{ml_ai}% AI</span>&nbsp;
            <span style="color:#475569;font-size:.8rem;">— {ml_lbl}</span>
          </div>
          <span style="color:#1e40af;font-size:.7rem;background:#1e3a5f;padding:.2rem .5rem;
            border-radius:5px;font-weight:700;">PRIMARY SIGNAL</span>
        </div>""", unsafe_allow_html=True)
    else:
      st.info("Fast heuristic analysis is active. Enable the pretrained classifier above for the slower ML ensemble.")

    # ── Watermark alert banner ────────────────────────────────────────────
    if wm_detected:
        st.markdown("""
        <div style="background:rgba(239,68,68,.12);border:1.5px solid rgba(239,68,68,.4);
          border-radius:12px;padding:.9rem 1.2rem;margin-bottom:1rem;display:flex;
          align-items:center;gap:.8rem;">
          <span style="font-size:1.5rem;">✦</span>
          <div>
            <div style="color:#fca5a5;font-weight:700;font-size:.9rem;">Gemini-style mark detected · origin unverified</div>
            <div style="color:#f87171;font-size:.8rem;margin-top:.1rem;">
              A visual match can be copied onto another image; this check cannot authenticate who added it.
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        if rr.get("possible_watermark_edit"):
            st.warning(
            f"Possible post-processing signal ({rr['watermark_edit_reason']}); "
            "this does not prove the watermark was added later."
            )

    g_col, v_col = st.columns([1,1], gap="large")

    with g_col:
        st.markdown(gauge_svg(ai_pct, color), unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align:center;margin-top:.3rem;">'
            f'<span class="verdict {vcls}">{verdict}</span>'
            f'<span class="conf-badge {ccls}">{conf} Confidence</span>'
            f'</div>', unsafe_allow_html=True)

    with v_col:
        heading("📊","Module Breakdown")
        radar = radar_img(ms)
        st.image(radar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)


    # Module score pills — dynamic, only show keys present in ms
    PILL_LABELS = {"ml":"ML Model","metadata":"Metadata","ela":"ELA","fft":"FFT",
                   "noise":"Noise","jpeg":"JPEG","face":"Face","watermark":"Watermark"}
    active_keys = [k for k in PILL_LABELS if k in ms]
    cols = st.columns(max(len(active_keys), 1))
    for i, k in enumerate(active_keys):
      with cols[i]: mod_pill(PILL_LABELS[k], ms[k]['raw'], ms[k]['weight'])

    st.markdown("<br>", unsafe_allow_html=True)
    heading("🔎","Evidence & Findings")

    # Group evidence
    high_ev = [e for e in rr["evidence"] if ev_class(e)=="high"]
    med_ev  = [e for e in rr["evidence"] if ev_class(e)=="med"]
    ok_ev   = [e for e in rr["evidence"] if ev_class(e)=="ok"]

    if high_ev:
        st.markdown('<div style="font-size:.75rem;font-weight:700;color:#f87171;letter-spacing:.08em;text-transform:uppercase;margin:.5rem 0 .25rem;">⚠ High Suspicion</div>', unsafe_allow_html=True)
        render_evidence(high_ev)
    if med_ev:
        st.markdown('<div style="font-size:.75rem;font-weight:700;color:#fb923c;letter-spacing:.08em;text-transform:uppercase;margin:.7rem 0 .25rem;">⚡ Medium Suspicion</div>', unsafe_allow_html=True)
        render_evidence(med_ev)
    if ok_ev:
        st.markdown('<div style="font-size:.75rem;font-weight:700;color:#34d399;letter-spacing:.08em;text-transform:uppercase;margin:.7rem 0 .25rem;">✓ Normal Indicators</div>', unsafe_allow_html=True)
        render_evidence(ok_ev)

# ─────────────────────────────────────────────────────────────────────────
# METADATA TAB
# ─────────────────────────────────────────────────────────────────────────
with t_meta:
    heading("🗂","EXIF Metadata")
    if mr["found"]:
        rows = [("Camera Model", mr.get("camera_model") or "—"),
                ("Lens",         mr.get("lens")         or "—"),
                ("Date Taken",   mr.get("date_taken")   or "—"),
                ("Software",     mr.get("software")     or "—"),
                ("GPS",          mr.get("gps")          or "—")]
        html = '<table style="width:100%;border-collapse:collapse;font-size:.85rem;">'
        html += '<tr><th style="color:#63b3ed;padding:.4rem .8rem;border-bottom:1px solid rgba(99,179,237,.15);text-align:left;">Field</th><th style="color:#63b3ed;padding:.4rem .8rem;border-bottom:1px solid rgba(99,179,237,.15);text-align:left;">Value</th></tr>'
        for lbl,val in rows:
            html += f'<tr><td style="color:#64748b;padding:.45rem .8rem;border-bottom:1px solid rgba(255,255,255,.04);font-size:.8rem;">{lbl}</td><td style="color:#cbd5e1;padding:.45rem .8rem;border-bottom:1px solid rgba(255,255,255,.04);font-family:JetBrains Mono,monospace;font-size:.8rem;">{val}</td></tr>'
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)
        if len(mr.get("raw_tags",{})) > 5:
            with st.expander("View all EXIF tags"):
                for k,v in list(mr["raw_tags"].items())[:60]:
                    st.caption(f"**{k}:** {v}")
    else:
        st.warning("No EXIF metadata found — common in AI-generated images.")
    st.markdown("<br>", unsafe_allow_html=True)
    pbar(mr["score"])
    render_evidence(mr["flags"])

# ─────────────────────────────────────────────────────────────────────────
# ELA TAB
# ─────────────────────────────────────────────────────────────────────────
with t_ela:
    e_img_col, e_met_col = st.columns([1,1], gap="large")
    with e_img_col:
        heading("🎨","Error Level Analysis Map")
        st.image(er["ela_map"], use_container_width=True)
    with e_met_col:
        heading("📈","Compression Variance")
        c1,c2 = st.columns(2)
        c1.metric("Mean Diff",   f"{er['mean_diff']:.1f}")
        c2.metric("Std Dev",     f"{er['std_diff']:.1f}")
        st.markdown("<br>", unsafe_allow_html=True)
        pbar(er["score"])
        st.markdown("<br>", unsafe_allow_html=True)
        render_evidence(er["flags"])
        st.info("ELA highlights areas of different compression levels. Real photos with varied textures or compression histories show varied ELA blocks. AI outputs usually exhibit uniformly low ELA variance.")

# ─────────────────────────────────────────────────────────────────────────
# FFT TAB
# ─────────────────────────────────────────────────────────────────────────
with t_fft:
    f_img_col, f_met_col = st.columns([1,1], gap="large")
    with f_img_col:
        heading("📡","FFT Frequency Spectrum")
        st.image(fr["spectrum_image"], use_container_width=True)
    with f_met_col:
        heading("📈","Frequency Metrics")
        c1,c2,c3 = st.columns(3)
        c1.metric("Grid Score",  f"{fr['grid_score']:.3f}")
        c2.metric("HF Ratio",    f"{fr['high_freq_ratio']:.3f}")
        c3.metric("Power Dev",   f"{fr.get('power_dev',0):.3f}")
        st.markdown("<br>", unsafe_allow_html=True)
        pbar(fr["score"])
        st.markdown("<br>", unsafe_allow_html=True)
        render_evidence(fr["flags"])
        st.info("Grid artifacts and high-frequency anomalies can indicate AI upsampling pipelines. A natural 1/f power spectrum deviation also signals generation artifacts.")

# ─────────────────────────────────────────────────────────────────────────
# NOISE TAB
# ─────────────────────────────────────────────────────────────────────────
with t_noise:
    n_img_col, n_met_col = st.columns([1,1], gap="large")
    with n_img_col:
        heading("🌊","Noise Heatmap")
        st.image(nr["noise_map_img"], use_container_width=True)
    with n_met_col:
        heading("📈","Noise Metrics")
        c1,c2,c3 = st.columns(3)
        c1.metric("Global σ",    f"{nr['global_sigma']:.2f}")
        c2.metric("Smooth %",    f"{nr['smoothness_pct']:.0f}%")
        c3.metric("σ Variance",  f"{nr['noise_variance']:.3f}")
        st.markdown("<br>", unsafe_allow_html=True)
        pbar(nr["score"])
        st.markdown("<br>", unsafe_allow_html=True)
        render_evidence(nr["flags"])
        st.info("Real camera sensors introduce photon noise (σ typically 3–12). AI generators produce unnaturally smooth images. High smooth-region % is a strong AI indicator.")

# ─────────────────────────────────────────────────────────────────────────
# JPEG TAB
# ─────────────────────────────────────────────────────────────────────────
with t_jpeg:
    heading("📦","Compression Analysis")
    j_rows = [("Format",            jr["file_format"]),
              ("File Size",         f"{jr['file_size_kb']:.1f} KB"),
              ("Est. Quality",      f"{jr['estimated_quality']} / 95"),
              ("Bytes / Pixel",     f"{jr['bytes_per_pixel']:.4f}"),
              ("Blocking Score",    f"{jr['blocking_score']:.3f}")]
    html = '<table style="width:100%;border-collapse:collapse;font-size:.85rem;">'
    html += '<tr><th style="color:#63b3ed;padding:.4rem .8rem;border-bottom:1px solid rgba(99,179,237,.15);text-align:left;">Metric</th><th style="color:#63b3ed;padding:.4rem .8rem;border-bottom:1px solid rgba(99,179,237,.15);text-align:left;">Value</th></tr>'
    for lbl,val in j_rows:
        html += f'<tr><td style="color:#64748b;padding:.45rem .8rem;border-bottom:1px solid rgba(255,255,255,.04);font-size:.8rem;">{lbl}</td><td style="color:#cbd5e1;padding:.45rem .8rem;border-bottom:1px solid rgba(255,255,255,.04);font-family:JetBrains Mono,monospace;font-size:.8rem;">{val}</td></tr>'
    html += '</table>'
    st.markdown(html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    pbar(jr["score"])
    render_evidence(jr["flags"])

# ─────────────────────────────────────────────────────────────────────────
# FACE TAB
# ─────────────────────────────────────────────────────────────────────────
with t_face:
    heading("👤","Face Analysis")
    if fc["face_detected"]:
        fa_col, fm_col = st.columns([1,1], gap="large")
        with fa_col:
            if fc["annotated_image"]:
                st.image(fc["annotated_image"], use_container_width=True,
                         caption=f"{fc['num_faces']} face(s) — MediaPipe FaceLandmarker")
        with fm_col:
            heading("📐","Geometry Metrics")
            met = fc.get("metrics") or {}
            if met:
                c1,c2 = st.columns(2)
                c1.metric("Eye Symmetry",    f"{met.get('eye_symmetry',0):.3f}")
                c2.metric("Ear Alignment",   f"{met.get('ear_alignment',0):.3f}")
                c1.metric("Face Proportion", f"{met.get('face_proportion',0):.3f}")
                c2.metric("Deepfake Prob",   f"{int((fc.get('deepfake_prob') or 0)*100)}%")
            st.markdown("<br>", unsafe_allow_html=True)
            pbar(fc["score"], color="#fb923c")
            render_evidence(fc["flags"])
    else:
        st.info("No face detected in this image — face geometry analysis skipped.")
        render_evidence(fc["flags"])

# ── Disclaimer ────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2.5rem;padding:1rem 1.4rem;background:rgba(255,255,255,.02);
  border-radius:12px;border:1px solid rgba(255,255,255,.05);">
  <p style="color:#374151;font-size:.78rem;margin:0;line-height:1.7;">
    <strong style="color:#4b5563;">⚠ Disclaimer:</strong>
    VisionForensics AI uses heuristic signal analysis — not a certified forensic tool.
    Results are probabilistic. Does not detect SynthID or C2PA watermarks.
    False positives and negatives are possible.
  </p>
  <p style="color:#1f2937;font-size:.72rem;margin:.4rem 0 0;">
    Planned: AI classifier · C2PA · ELA · Lighting coherence · Object consistency · DeepFake v2
  </p>
</div>""", unsafe_allow_html=True)
