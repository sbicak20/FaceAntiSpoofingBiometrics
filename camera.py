import cv2
import joblib
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
        preprocessed_face = preprocessed_face.flatten()

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
