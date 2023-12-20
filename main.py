import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage import feature
from tqdm import tqdm
import joblib
# Function to extract HOG features from an image
def extract_hog_features(image_path, min_size=(32, 32)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    if image.shape[0] < min_size[0] or image.shape[1] < min_size[1]:
        image = cv2.resize(image, min_size, interpolation=cv2.INTER_LINEAR)
    hog_features = feature.hog(image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    return hog_features

# Function to extract HOG features from a batch of images
def extract_hog_features_batch(image_paths, min_size=(32, 32)):
    features_list = []
    for image_path in image_paths:
        features = extract_hog_features(image_path, min_size)
        if features is not None:
            features_list.append(features)
    return features_list

# Path to the folder containing spoofed and live images
spoof_folder_path = r"C:\Users\sebas\Documents\FaceSpoofsDatasets\zips\CelebA_Spoof_train\Data\spoofedimagestest2"
live_folder_path = r"C:\Users\sebas\Documents\FaceSpoofsDatasets\zips\CelebA_Spoof_train\Data\liveimagestest2"

# List to store image paths and labels
spoof_image_paths = []
live_image_paths = []

# Iterate over files in the folder with tqdm for the progress bar
for filename in tqdm(os.listdir(spoof_folder_path), desc="Loading spoof images", unit="image"):
    if filename.endswith(".png"):
        # Append image path
        spoof_image_path = os.path.join(spoof_folder_path, filename)
        spoof_image_paths.append(spoof_image_path)

for filename in tqdm(os.listdir(live_folder_path), desc="Loading live images", unit="image"):
    if filename.endswith(".png"):
        # Append image path
        live_image_path = os.path.join(live_folder_path, filename)
        live_image_paths.append(live_image_path)

# Labels: 1 for spoofed, 0 for live
spoof_labels = [1] * len(spoof_image_paths)
live_labels = [0] * len(live_image_paths)

# Combine spoof and live paths and labels
image_paths = spoof_image_paths + live_image_paths
labels = spoof_labels + live_labels

# Shuffle the data
#combined_data = list(zip(image_paths, labels))
#np.random.shuffle(combined_data)
#image_paths, labels = zip(*combined_data)

# Use a generator for feature extraction
def feature_generator(image_paths, labels, batch_size=32):
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        batch_features = [extract_hog_features(path) for path in batch_paths]
        valid_features = [feature for feature in batch_features if feature is not None]
        max_size = max(len(feature) for feature in valid_features)
        padded_features = [np.pad(feature, (0, max_size - len(feature)), mode='constant') for feature in valid_features]
        yield np.vstack(padded_features), np.array(batch_labels)


# Split the data into train and test sets
train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2,
                                                                                  random_state=42)

# Initialize SVM model
model = SVC(kernel='linear', C=1.0)

# Train the model
batch_size = 32
num_epochs = 1  # You can experiment with the number of epochs
for epoch in range(num_epochs):
    train_generator = feature_generator(train_image_paths, train_labels, batch_size=batch_size)
    for X_train_batch, y_train_batch in tqdm(train_generator, total=len(train_image_paths) // batch_size,
                                             desc=f"Epoch {epoch + 1}/{num_epochs}"):
        # Debugging: Print shapes before training
        print()
        print("X_train_batch shape before training:", X_train_batch.shape)
        print("y_train_batch shape before training:", y_train_batch.shape)

        model.fit(X_train_batch, y_train_batch)

# Test the model
test_generator = feature_generator(test_image_paths, test_labels, batch_size=batch_size)
all_predictions = []
for X_test_batch, y_test_batch in test_generator:
    # Debugging: Print shapes before testing
    print()
    print("X_test_batch shape before testing:", X_test_batch.shape)
    print("y_test_batch shape before testing:", y_test_batch.shape)

    predictions = model.predict(X_test_batch)
    all_predictions.extend(predictions)

# Evaluate the model
accuracy = accuracy_score(test_labels, all_predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

model_filename = "trained_model.joblib"
joblib.dump(model, model_filename)
