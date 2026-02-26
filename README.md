🧠 CrowdSense
Crowd Density Classification using HOG and Machine Learning
📌 Project Overview
CrowdSense is a classical machine learning–based system designed to classify images or video frames into Crowded and Non-Crowded categories.

The system uses:

HOG (Histogram of Oriented Gradients) for feature extraction

SVM and other ML classifiers for classification

The goal is to build a lightweight and efficient crowd monitoring solution without deep learning.

🎯 Problem Statement
To automatically detect and classify crowd density in surveillance images using classical computer vision and machine learning techniques.

🏗️ System Pipeline
Input Image / Frame
        ↓
Preprocessing (Resize + Grayscale + CLAHE)
        ↓
HOG Feature Extraction (8100 features)
        ↓
Machine Learning Classifier (SVM)
        ↓
Crowded / Non-Crowded Prediction
⚙️ Technologies Used
Python

OpenCV

NumPy

Scikit-learn

Matplotlib

🔍 Feature Extraction
HOG Parameters Used:
Image Size: 128 × 128

Pixels per Cell: 8 × 8

Cells per Block: 2 × 2

Orientation Bins: 9

Block Normalization: L2-Hys

Feature Vector Size: 8100 per image

HOG captures local edge orientation patterns which represent structural information of crowded scenes.

🤖 Machine Learning Models Tested
K-Nearest Neighbors (KNN)

Random Forest

Gradient Boosting

Support Vector Machine (SVM)

Final Selected Model:
✔ RBF-SVM (Best accuracy observed)

📊 Model Evaluation
Evaluation metrics used:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

🚀 How to Run the Project
1️⃣ Install Dependencies
pip install opencv-python numpy scikit-learn matplotlib scikit-image
2️⃣ Run Training
python train_model.py
3️⃣ Run Prediction
python predict.py
📁 Project Structure
CrowdSense/
│
├── dataset/
│   ├── crowded/
│   └── non_crowded/
│
├── hog_feature_extraction.py
├── train_model.py
├── predict.py
├── hog_visualization.py
├── model.pkl
└── README.md
🔬 Technical Highlights
Uses handcrafted gradient-based features (HOG)

Handles high-dimensional feature space (8100 features)

SVM learns feature importance automatically

Robust to lighting variations using CLAHE

Computationally efficient compared to deep learning

📈 Key Contributions
Designed preprocessing pipeline (Resize + CLAHE)

Configured optimal HOG parameters

Compared multiple ML classifiers

Selected best-performing model

Implemented complete end-to-end system

⚠️ Limitations
Binary classification only

Performance depends on dataset quality

Not optimized for extreme occlusion

🔮 Future Improvements
Multi-level crowd density classification

Real-time video deployment

Integration with IoT alert systems

Comparison with CNN-based approaches

📌 Conclusion
CrowdSense demonstrates that classical computer vision techniques combined with machine learning can effectively perform crowd density classification in a computationally efficient manner.

