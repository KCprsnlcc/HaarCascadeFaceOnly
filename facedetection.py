import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    """Detect the most prominent face in the input image and draw a rectangle around it."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Select the first detected face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return img

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_with_face = detect_face(frame)

        cv2.imshow('Face Detection', frame_with_face)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
