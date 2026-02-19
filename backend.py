import cv2
import joblib
import requests
from skimage.feature import hog

# ======================================================
# 🚨 Telegram Configuration
# ======================================================

BOT_TOKEN = "8424792466:AAFlQBOQdkHA_MdBL14lmOKao-zQoQaB1Jo"
CHAT_ID ="5377934639"

def send_telegram_alert():
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": "🚨 IMPORTANT ALERT: Crowd density has exceeded safe limits at the monitored location. Immediate attention required."
    }
    requests.post(url, data=payload)

# ======================================================
# 🧠 Load Model & Scaler (ONCE)
# ======================================================
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ======================================================
# 🎥 Backend Video Processing Generator
# ======================================================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    crowded_frames = 0
    frames_in_second = 0
    second_count = 0

    ALERT_THRESHOLD = 10
    alert_sent = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ---------- Preprocessing ----------
        resized = cv2.resize(frame, (128, 128))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        enhanced = clahe.apply(gray)

        # ---------- HOG ----------
        features = hog(
            enhanced,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        features = scaler.transform([features])
        pred = model.predict(features)[0]

        # ---------- Crowd Logic ----------
        if pred == 0:
            crowded_frames += 1
            frames_in_second += 1
        else:
            frames_in_second += 1
            alert_sent = False

        # ---------- Telegram Alert ----------
        if crowded_frames >= ALERT_THRESHOLD and not alert_sent:
            send_telegram_alert()
            alert_sent = True

        # ---------- Per-second aggregation ----------
        density_update = None
        if frames_in_second >= fps:
            second_count += 1
            density_update = (second_count, crowded_frames)

            crowded_frames = 0
            frames_in_second = 0

        # Yield everything frontend needs
        yield frame, pred, density_update

    cap.release()