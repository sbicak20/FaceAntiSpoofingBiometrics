import os

import cv2
import numpy as np
import pickle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import joblib
import matplotlib.pyplot as plt
from helpers.imagePath import SaveImagesScikit, SaveImages

input_dir = r"C:\Users\sebas\Documents\FaceSpoofsDatasets\zips\CelebA_Spoof_train\Data"
output_folder = r"C:\Users\sebas\Documents\FaceSpoofsDatasets\zips\CelebA_Spoof_train\Data\processedImages"
categories = ["spoofedimagestest4", "liveimagestest3"]

def ShowImage(image):
    plt.imshow(image, cmap='gray')
    plt.title('Processed Image')
    plt.axis('on')
    plt.show()

data = []
labels = []
features = []
for category_index, category in enumerate(categories):
    hog_images = []
    greyscale_images = []
    resized_images = []
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir,category,file)
        # greyscale
        img = imread(img_path, as_gray=True)
        #ShowImage(img)
        greyscale_images.append(img)

        #resized
        img = resize(img,(64,64))
        #ShowImage(img)
        resized_images.append(img)

        features_per_image, hog_image = hog(img, orientations=8, pixels_per_cell=(10, 10), cells_per_block=(3, 3),
                                            visualize=True)
        features.append(features_per_image.flatten())
        #data.append(img.flatten())
        labels.append(category_index)

        #HOG
        #ShowImage(hog_image)
        hog_images.append(hog_image)

    SaveImages(hog_images, output_folder, category + r"HOG")
    SaveImages(hog_images, output_folder, category + r"Greyscale")
    SaveImages(hog_images, output_folder, category + r"Resized")

#data = np.asarray(data)

data = np.asarray(features)
lables = np.asarray(labels)

# split the data

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train model
classifier = SVC()
classifier.fit(x_train, y_train)

# test model
y_prediction = classifier.predict(x_test)
score = accuracy_score(y_prediction, y_test)
print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(classifier, open('./faceantispoofmodel1500withHOG.p', 'wb'))
joblib.dump(classifier, 'faceantispoofmodel1500withHOG.joblib')
