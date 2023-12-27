import os
import numpy as np
import pickle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


input_dir = r"C:\Users\sebas\Documents\FaceSpoofsDatasets\zips\CelebA_Spoof_train\Data"
categories = ["spoofedimagestest2", "liveimagestest2"]

data = []
labels = []
for category_index, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir,category,file)
        img = imread(img_path, as_gray=True)
        img = resize(img,(32,32))
        data.append(img.flatten())
        labels.append(category_index)

data = np.asarray(data)
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

pickle.dump(classifier, open('./faceantispoofmodel1000.p', 'wb'))
joblib.dump(classifier, 'faceantispoofmodel1000.joblib')
