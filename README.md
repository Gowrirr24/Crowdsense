# 🧠 CrowdSense  
## Crowd Density Classification using HOG + SVM  

---

## 📌 Overview  

CrowdSense is a classical machine learning–based computer vision system designed to classify images or video frames into:

- ✅ Crowded  
- ❌ Non-Crowded  

The system uses **Histogram of Oriented Gradients (HOG)** for feature extraction and **Support Vector Machine (SVM)** for classification.

This project demonstrates that handcrafted features combined with classical machine learning techniques can effectively perform crowd density detection without deep learning.

---

## 🎯 Objective  

To design a computationally efficient crowd monitoring system that:

1. Extracts structural edge-based features  
2. Learns discriminative crowd patterns  
3. Classifies scenes based on density  
4. Provides reliable prediction performance  

---

## 🏗️ System Architecture  

1. Input Image / Frame  
2. Preprocessing (Resize + Grayscale + CLAHE)  
3. HOG Feature Extraction (8100 features per image)  
4. Machine Learning Model (SVM)  
5. Crowded / Non-Crowded Prediction  

---

## 🔍 Feature Extraction – HOG  

The system uses **Histogram of Oriented Gradients (HOG)** to convert images into numerical feature vectors.

### HOG Configuration:

- Image Size: 128 × 128  
- Pixels per Cell: 8 × 8  
- Cells per Block: 2 × 2  
- Orientation Bins: 9  
- Block Normalization: L2-Hys  
- Feature Vector Length: 8100  

HOG captures local gradient orientation distributions, which effectively represent crowd structures and dense human patterns.

---

## 🤖 Machine Learning Models Evaluated  

The following classifiers were trained and compared:

1. K-Nearest Neighbors (KNN)  
2. Random Forest  
3. Gradient Boosting  
4. Support Vector Machine (SVM)  

### ✅ Final Selected Model:

**RBF Kernel SVM** (achieved best performance)

---

## 📊 Evaluation Metrics  

Model performance was evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

---

## 🧪 Technical Highlights  

- High-dimensional feature representation (8100 features per image)  
- Automatic feature weighting through SVM  
- Illumination robustness using CLAHE  
- Margin-based classification in high-dimensional space  
- Lightweight alternative to deep learning  

---

## 📁 Project Structure  
CrowdSense/
│
├── dataset/
│ ├── crowded/
│ └── non_crowded/
│
├── hog_feature_extraction.py
├── train_model.py
├── predict.py
├── hog_visualization.py
├── model.pkl
└── README.md


---

## 🔮 Future Enhancements  

1. Multi-level crowd density classification  
2. Real-time video stream integration  
3. Automated alert triggering system  
4. Comparative study with CNN-based models  
