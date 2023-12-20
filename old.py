import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage import feature
from tqdm import tqdm
import joblib


# Function to extract HOG features from an image
def extract_hog_features(image_path, min_size=(32, 32)):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is successfully loaded
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # Check if the image is too small, resize if necessary
    if image.shape[0] < min_size[0] or image.shape[1] < min_size[1]:
        image = cv2.resize(image, min_size, interpolation=cv2.INTER_LINEAR)

    # Extract HOG features
    hog_features = feature.hog(image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    return hog_features




# Path to the folder containing spoofed images
spoof_folder_path = r"C:\Users\sebas\Documents\FaceSpoofsDatasets\zips\CelebA_Spoof_train\Data\spoofedimages2"
live_folder_path = r"C:\Users\sebas\Documents\FaceSpoofsDatasets\zips\CelebA_Spoof_train\Data\liveimages2"

# List to store image paths and labels
image_paths = []
labels = []


# Iterate over files in the folder with tqdm for the progress bar
for filename in tqdm(os.listdir(spoof_folder_path), desc="Processing spoof images", unit="image"):
    if filename.endswith(".png"):
        # Append image path
        image_path = os.path.join(spoof_folder_path, filename)
        # Extract features
        features = extract_hog_features(image_path)

        # Check if features are not None
        if features is not None:
            # Append image path and label
            image_paths.append(image_path)
            labels.append(1)  # 1 for spoofed

for filename in tqdm(os.listdir(live_folder_path), desc="Processing live images", unit="image"):
    if filename.endswith(".png"):
        # Append image path
        image_path = os.path.join(live_folder_path, filename)
        # Extract features
        features = extract_hog_features(image_path)

        # Check if features are not None
        if features is not None:
            # Append image path and label
            image_paths.append(image_path)
            labels.append(0)  # 1 for spoofed
# Feature extraction
features = [extract_hog_features(path) for path in tqdm(image_paths, desc="Extracting features", unit="image") if extract_hog_features(path) is not None]

# we can remove the loaded images and lables after the extraction is done


# Find the maximum feature size
max_size = max(len(feature) for feature in features)

# Pad or truncate features to have a consistent size
features = [np.pad(feature, (0, max_size - len(feature)), mode='constant') for feature in features]

# Convert the list of 1D arrays to a 2D array
features = np.vstack(features)

# Reshape features to ensure they are 2D
features = features.reshape(len(features), -1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
del image_paths
del labels
# Train an SVM model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

model_filename = "trained_model_old.joblib"
joblib.dump(model, model_filename)
