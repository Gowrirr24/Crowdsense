import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# Load Feature Data
# -------------------------------
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

print("✅ Features Loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("✅ Train-Test Split Done")

# -------------------------------
# Feature Scaling
# -------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("✅ Feature Scaling Done")

# -------------------------------
# Train SVM Classifier
# -------------------------------
model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("🎉 SVM Training Completed!")

# -------------------------------
# Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Save Model + Scaler
# -------------------------------
joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n💾 Model saved as svm_model.pkl")
print("💾 Scaler saved as scaler.pkl")
