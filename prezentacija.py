import os
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from skimage import exposure, restoration
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt


input_dir = r"C:\Users\sebas\Documents\CelebA_Spoof\Data"
categories = ["batchestrainspoof", "batchestrainlive"]
batchfolders = ["b1","b2"]

classifier = SVC()
for batchfolder in batchfolders:
    data = []
    labels = []

    for category_index, category in enumerate(categories):
        features = []

        for file in os.listdir(os.path.join(input_dir, category, batchfolder)):
            img_path = os.path.join(input_dir, category, batchfolder, file)

            # Display images
            fig, axes = plt.subplots(1, 5, figsize=(12, 3))
            ax = axes.ravel()

            #greyscale
            img = imread(img_path, as_gray=True)
            ax[0].imshow(img, cmap=plt.cm.gray)
            ax[0].set_title('Original Image')

            #resize
            img = resize(img, (64, 64))
            ax[1].imshow(img, cmap=plt.cm.gray)
            ax[1].set_title('Resized Image')

            #denoising
            img_denoised = restoration.denoise_bilateral(img, sigma_color=0.05, sigma_spatial=15)
            ax[2].imshow(img_denoised, cmap=plt.cm.gray)
            ax[2].set_title('Denoised Image')

            #contrast
            img_adaptive_equalized = exposure.equalize_adapthist(img_denoised, clip_limit=0.03)
            ax[3].imshow(img_adaptive_equalized, cmap=plt.cm.gray)
            ax[3].set_title('Adaptive Equalized Image')

            #feature extraction, HOG
            features_per_image, hog_image = hog(img_adaptive_equalized, orientations=8, pixels_per_cell=(10, 10), cells_per_block=(3, 3),
                                        visualize=True)
            features.append(features_per_image.flatten())
            ax[4].imshow(hog_image, cmap=plt.cm.gray)
            ax[4].set_title('HOG Image')

            labels.append(category_index)

            for a in ax:
                a.axis('off')

            plt.tight_layout()
            plt.show()

        data.extend(features)

    data = np.asarray(data)
    labels = np.asarray(labels)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    classifier.fit(x_train, y_train)

    y_prediction = classifier.predict(x_test)

    accuracy = accuracy_score(y_prediction, y_test)
    precision = precision_score(y_test, y_prediction, average='binary')
    recall = recall_score(y_test, y_prediction, average='binary')
    f1 = f1_score(y_test, y_prediction, average='binary')

    print(f'{batchfolder}: {accuracy * 100}% of samples were correctly classified.')
    model_name = f'faceantispoofmodelafter{batchfolder}.joblib'
    joblib.dump(classifier, model_name)

    print(f'Accuracy Score: {accuracy}')
    print(f'Precision Score: {precision}')
    print(f'Recall Score: {recall}')
    print(f'F1 Score: {f1}')

    #Plotting Accuracy and Precision
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(['Accuracy'], [accuracy], color=color, marker='o', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Precision', color=color)
    ax2.plot(['Precision'], [precision], color=color, marker='o', label='Precision')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Accuracy and Precision')
    plt.show()

    # Plotting Recall and F1 Score
    fig, ax1 = plt.subplots()

    color = 'tab:green'
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Recall', color=color)
    ax1.plot(['Recall'], [recall], color=color, marker='o', label='Recall')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:purple'
    ax2.set_ylabel('F1 Score', color=color)
    ax2.plot(['F1 Score'], [f1], color=color, marker='o', label='F1 Score')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Recall and F1 Score')
    plt.show()




# Display the original, denoised, equalized, and adaptive equalized images
            #fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            #ax = axes.ravel()

            #ax[0].imshow(img, cmap=plt.cm.gray)
            #ax[0].set_title('Original Image')

            #ax[1].imshow(img_denoised, cmap=plt.cm.gray)
            #ax[1].set_title('Denoised Image')

            #ax[2].imshow(img_adaptive_equalized, cmap=plt.cm.gray)
            #ax[2].set_title('Adaptive Equalized Image')
#

# ax[3].imshow(hog_image, cmap=plt.cm.gray)
# ax[3].set_title('Hog Image')

# for a in ax:
#    a.axis('off')

# plt.tight_layout()
# plt.show()