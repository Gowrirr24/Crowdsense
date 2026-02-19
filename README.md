CrowdSense – Crowd Scene Classification and Alert System
📌 Project Description
CrowdSense is a real‑time crowd monitoring system that analyzes video streams to classify scenes as crowded or non‑crowded using classical machine‑learning techniques. The system uses HOG feature extraction with an RBF‑kernel Support Vector Machine and sends Telegram alerts when crowd density exceeds a predefined threshold.

🎯 Objectives
Detect crowded and non‑crowded scenes from video

Compare multiple machine‑learning models

Perform real‑time crowd monitoring

Trigger alerts for high crowd density

🧠 Technologies Used
Python

OpenCV

Scikit‑learn

Streamlit

Telegram Bot API

⚙️ Methodology
Video frames are extracted from input video

Frames are preprocessed using grayscale conversion and CLAHE

HOG features are extracted from each frame

Multiple ML models are trained and evaluated

RBF‑SVM is selected as the final model

Continuous crowded frames trigger Telegram alerts

📊 Models & Accuracy
RBF‑SVM – 90% (Selected)

Random Forest – 87%

Gradient Boosting – 82%

KNN – 48%

🚨 Alert System
When the number of continuous crowded frames exceeds a fixed threshold, an automated alert is sent to users using the Telegram Bot API for quick response.

📁 Project Files
app.py – Streamlit frontend

backend.py – Video processing logic

hog_extract.py – Feature extraction

train_*.py – Model training scripts

*.pkl – Trained model files

✅ Conclusion
CrowdSense provides a lightweight and efficient solution for crowd monitoring using classical machine‑learning methods. The system is suitable for public safety applications and can be extended for real‑time CCTV deployment.
