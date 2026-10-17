import cv2

# Load OpenCV's built-in face detector (a Haar cascade)
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open the webcam (0 = the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open the webcam")
    raise SystemExit

print("Webcam running. Press q to quit.")

while True:
    ret, frame = cap.read()          # grab one frame
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # detector works on grayscale

    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()