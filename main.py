import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage import feature
from tqdm import tqdm
import joblib
import sys
import matplotlib.pyplot as plt
sys.path.append("./helpers")
from helpers.imagePath import GetLiveImagePaths, GetSpoofImagePaths, SaveImages
from helpers.preprocessimages import PreprocessImage, ConvertToGreyscale
from helpers.haarfeature import GetFaceImage



# 1. getting images and labels
spoof_image_paths = GetSpoofImagePaths()
live_image_paths = GetLiveImagePaths()
# Labels: 1 for spoofed, 0 for live
spoof_labels = [1] * len(spoof_image_paths)
live_labels = [0] * len(live_image_paths)
# Combine spoof and live paths and labels
image_paths = spoof_image_paths + live_image_paths
labels = spoof_labels + live_labels

# 2. preprocess spoof images to be greyscale
grey_spoof_images = []
for spoof_image_path in spoof_image_paths:
    grey_spoof_images.append(ConvertToGreyscale(spoof_image_path))

SaveImages(grey_spoof_images, r"\spoof\greyscale_spoof_images")

# 3. get haar features
face_spoof_images = []
for grey_spoof_image in grey_spoof_images:
    face_spoof_from_grey = GetFaceImage(grey_spoof_image)
    if face_spoof_from_grey is not None:
        face_spoof_images.append(face_spoof_from_grey)
SaveImages(face_spoof_images, r"\spoof\face_spoof_images")

# 4. preprocess images
preprocessed_face_images = []
#for face_spoof_image in face_spoof_images:
preprocessed_face_images.append(PreprocessImage(r"C:\Users\sebas\Documents\FaceSpoofsDatasets\zips\CelebA_Spoof_train\Data\realdata\spoof\face_spoof_images\output_image_0.png"))
SaveImages(preprocessed_face_images, r"\spoof\preprocessed_face_spoof_images")
# 5. use pca

# 6. train model

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
