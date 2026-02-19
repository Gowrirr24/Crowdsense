import streamlit as st
import cv2
import tempfile
import pandas as pd

# 🔗 IMPORT BACKEND
from backend import process_video

# ======================================================
# 🌌 DARK NEON GLASSMORPHISM THEME
# ======================================================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(0,245,255,0.12), transparent 40%),
        radial-gradient(circle at bottom right, rgba(255,0,204,0.12), transparent 40%),
        linear-gradient(135deg, #05070f, #020617, #05070f) !important;
    color: #e5e7eb !important;
}

.stApp {
    background: linear-gradient(135deg,
        rgba(15,23,42,0.75),
        rgba(2,6,23,0.85)) !important;
    backdrop-filter: blur(14px);
}

.block-container,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stFooter"] {
    background: transparent !important;
}

.block-container {
    background: rgba(15,23,42,0.45) !important;
    border-radius: 22px;
    padding: 2rem;
    backdrop-filter: blur(18px);
    box-shadow: 0 0 25px rgba(0,245,255,0.12);
}

h1 {
    color: #00f5ff !important;
    text-align: center;
    font-weight: 800;
    text-shadow: 0 0 12px #00f5ff;
}

.stButton > button {
    background: linear-gradient(
        135deg,
        rgba(255,0,204,0.25),
        rgba(51,51,255,0.25)
    ) !important;
    color: white !important;
    font-size: 18px;
    padding: 12px 32px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(12px);
    box-shadow: 0 0 18px rgba(255,0,204,0.6);
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 🎛️ UI
# ======================================================
st.markdown("<h1>🚦 CrowdSense – Smart Crowd Monitoring</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Crowded / Non‑Crowded classification with per‑second density trend</p>",
    unsafe_allow_html=True
)

uploaded_video = st.file_uploader(
    "📁 Upload Video File",
    type=["mp4", "avi", "mov"]
)

start = st.button("▶ Start Classification")

# ======================================================
# 🎥 FRONTEND + BACKEND INTEGRATION
# ======================================================
if uploaded_video and start:

    # Save uploaded video temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())

    frame_placeholder = st.empty()
    graph_placeholder = st.empty()

    seconds = []
    density_per_second = []

    # 🔁 Call backend generator
    for frame, pred, density_update in process_video(temp_file.name):

        # ----- Classification overlay -----
        label = "Crowded" if pred == 0 else "Non-Crowded"
        color = (0, 0, 255) if pred == 0 else (0, 255, 0)

        cv2.putText(frame, label, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # ----- Graph update (per second) -----
        if density_update:
            sec, count = density_update
            seconds.append(sec)
            density_per_second.append(count)

            df = pd.DataFrame({
                "Second": seconds,
                "Crowd Density": density_per_second
            })

            graph_placeholder.line_chart(df.set_index("Second"))

    st.success("✅ Video processing completed")