import os
import cv2

# Input and Output Paths
INPUT_DIR = "dataset"
OUTPUT_DIR = "preprocessed_dataset"

CLASSES = ["crowded", "non_crowded"]
IMAGE_SIZE = (128, 128)

# Create output folders
for cls in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def preprocess_image(img):
    # Resize
    img = cv2.resize(img, IMAGE_SIZE)

    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE
    enhanced = clahe.apply(gray)

    return enhanced

# Process all images
for cls in CLASSES:
    input_path = os.path.join(INPUT_DIR, cls)
    output_path = os.path.join(OUTPUT_DIR, cls)

    for file in os.listdir(input_path):
        img_path = os.path.join(input_path, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        processed = preprocess_image(img)

        # Save preprocessed image
        cv2.imwrite(os.path.join(output_path, file), processed)

    print(f"✅ Finished preprocessing: {cls}")

print("🎉 STEP‑1 Completed: All images saved in preprocessed_dataset/")
