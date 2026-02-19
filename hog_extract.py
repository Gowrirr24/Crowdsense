import os
import cv2
import numpy as np
from skimage.feature import hog

INPUT_DIR = "preprocessed_dataset"
CLASSES = ["crowded", "non_crowded"]

X = []
y = []

for label, cls in enumerate(CLASSES):

    folder = os.path.join(INPUT_DIR, cls)

    for file in os.listdir(folder):

        img_path = os.path.join(folder, file)

        # Read grayscale image
        img = cv2.imread(img_path, 0)

        if img is None:
            continue

        # Extract HOG features
        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            block_norm="L2-Hys"
        )

        X.append(features)
        y.append(label)

print("✅ Step‑2 Completed: HOG Features Extracted!")

X = np.array(X)
y = np.array(y)

# Save feature vectors
np.save("X_features.npy", X)
np.save("y_labels.npy", y)

print("🎉 Saved: X_features.npy and y_labels.npy")
