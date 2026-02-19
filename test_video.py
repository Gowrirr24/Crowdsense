import cv2
import joblib
import numpy as np
import requests
from skimage.feature import hog

# 🔹 Telegram config
BOT_TOKEN = "8424792466:AAFlQBOQdkHA_MdBL14lmOKao-zQoQaB1Jo"
CHAT_ID ="5377934639"

def send_telegram_alert():
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": "🚨 IMPORTANT ALERT: Crowd density has exceeded safe limits at the monitored location. Immediate attention required."
    }
    requests.post(url, data=payload)

# Load model & scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

print("Model and Scaler Loaded Successfully")

cap = cv2.VideoCapture("crowded.mp4")

if not cap.isOpened():
    print("Error opening video")
    exit()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# 🔹 Alert logic
crowd_count = 0
ALERT_THRESHOLD = 10
alert_sent = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Preprocessing ---
    frame_resized = cv2.resize(frame, (128,128))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)

    # --- HOG ---
    features = hog(enhanced,
                   orientations=9,
                   pixels_per_cell=(8,8),
                   cells_per_block=(2,2),
                   block_norm='L2-Hys')

    features = scaler.transform([features])

    # --- Prediction ---
    pred = model.predict(features)[0]

    label = "Crowded" if pred == 0 else "Non-Crowded"
    color = (0,0,255) if pred == 0 else (0,255,0)

    # 🔹 Crowd counter
    if pred == 0:
        crowd_count += 1
    else:
        crowd_count = 0
        alert_sent = False

    # 🔹 Trigger Telegram alert
    if crowd_count >= ALERT_THRESHOLD and not alert_sent:
        send_telegram_alert()
        alert_sent = True

    cv2.putText(frame, label,
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2)

    cv2.imshow("Crowd Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
