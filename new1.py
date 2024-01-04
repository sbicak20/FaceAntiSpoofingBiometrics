import os

import cv2
import numpy as np
import pickle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from skimage import exposure, restoration
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt

def ShowImage(image):
    plt.imshow(image, cmap='gray')
    plt.title('Processed Image')
    plt.axis('on')
    plt.show()


input_dir = r"C:\Users\sebas\Documents\FaceSpoofsDatasets\allzips\CelebASpoofAllZips\CelebA_Spoof\Data"
categories = ["batchestrainspoof", "batchestrainlive"]
batchfolders = ["b1","b2","b3","b4","b5","b6","b7","b8","b9","b10"]

classifier = SVC()

for batchfolder in batchfolders:
    data = []
    labels = []

    for category_index, category in enumerate(categories):
        features = []

        for file in os.listdir(os.path.join(input_dir, category, batchfolder)):
            img_path = os.path.join(input_dir, category, batchfolder, file)
            img = imread(img_path, as_gray=True)
            img = resize(img, (64, 64))

            # Denoising using bilateral filter
            img_denoised = restoration.denoise_bilateral(img, sigma_color=0.05, sigma_spatial=15)

            # Histogram equalization
            #img_equalized = exposure.equalize_hist(img_denoised)

            # Local contrast enhancement using adaptive histogram equalization
            img_adaptive_equalized = exposure.equalize_adapthist(img_denoised, clip_limit=0.03)

            # Display the original, denoised, equalized, and adaptive equalized images
            #fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            #ax = axes.ravel()

            #ax[0].imshow(img, cmap=plt.cm.gray)
            #ax[0].set_title('Original Image')

            #ax[1].imshow(img_denoised, cmap=plt.cm.gray)
            #ax[1].set_title('Denoised Image')

            #ax[2].imshow(img_equalized, cmap=plt.cm.gray)
            #ax[2].set_title('Equalized Image')

            #ax[3].imshow(img_adaptive_equalized, cmap=plt.cm.gray)
            #ax[3].set_title('Adaptive Equalized Image')
#
            #for a in ax:
            #    a.axis('off')

            #plt.tight_layout()
            #plt.show()

            features_per_image, _ = hog(img_adaptive_equalized, orientations=8, pixels_per_cell=(10, 10), cells_per_block=(3, 3),
                                        visualize=True)
            features.append(features_per_image.flatten())
            labels.append(category_index)

        data.extend(features)

    data = np.asarray(data)
    labels = np.asarray(labels)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    classifier.fit(x_train, y_train)

    y_prediction = classifier.predict(x_test)
    score = accuracy_score(y_prediction, y_test)

    print(f'{batchfolder}: {score * 100}% of samples were correctly classified.')
    model_name = f'faceantispoofmodelafter{batchfolder}.joblib'
    joblib.dump(classifier, model_name)

    # statistics
    conf_matrix = confusion_matrix(y_test, y_prediction)

    false_acceptances = conf_matrix[1, 0]  # Spoof (class 1) predicted as Live (class 0)
    false_rejections = conf_matrix[0, 1]  # Live (class 0) predicted as Spoof (class 1)

    total_impostor_attempts = np.sum(conf_matrix[1, :])
    total_genuine_attempts = np.sum(conf_matrix[0, :])

    far = (false_acceptances / total_impostor_attempts) * 100
    frr = (false_rejections / total_genuine_attempts) * 100

    print(f'FAR: {far:.2f}%')
    print(f'FRR: {frr:.2f}%')
    # AUC
    y_score = classifier.decision_function(x_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    auc = roc_auc_score(y_test, y_score)
    print(f'AUC: {auc:.2f}')