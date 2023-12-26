import cv2

def GetFaceImage(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = image[y:y+h, x:x+w]
        return face_roi
    else:
        return None

# Example usage:
#image_path = "path/to/your/preprocessed_image.jpg"
#face_image = detect_face(image_path)