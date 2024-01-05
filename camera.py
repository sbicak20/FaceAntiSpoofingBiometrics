import cv2
import joblib
from skimage import restoration
from skimage.exposure import exposure
from skimage.feature import hog
from skimage.transform import resize

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

loaded_model = joblib.load("faceantispoofmodelafterb10.joblib")

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=1, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_region = gray_frame[y:y+h, x:x+w]

        preprocessed_face = resize(face_region, (64, 64), anti_aliasing=True)
        img_denoised = restoration.denoise_bilateral(preprocessed_face, sigma_color=0.05, sigma_spatial=15)
        img_adaptive_equalized = exposure.equalize_adapthist(img_denoised, clip_limit=0.03)
        features_per_image, _ = hog(img_adaptive_equalized, orientations=8, pixels_per_cell=(10, 10),
                                    cells_per_block=(3, 3),
                                    visualize=True)
        preprocessed_face.extend(features_per_image.flatten())
        data = preprocessed_face.reshape(1, -1)

        prediction = loaded_model.predict(data)

        if prediction == 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
