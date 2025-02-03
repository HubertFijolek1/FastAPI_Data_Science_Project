import cv2

def detect_faces(image_path: str):
    """
    Detects faces in an image using Haar cascades.
    Returns list of bounding boxes (x, y, w, h).
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    return faces
