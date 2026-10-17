import cv2
from deepface import DeepFace

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
frame_count = 0
label_text = ""

print("Webcam running. Press q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        # Run the emotion model only on every 5th frame to stay smooth
        if frame_count % 5 == 0:
            face_img = frame[y:y + h, x:x + w]
            result = DeepFace.analyze(
                face_img,
                actions=["emotion"],
                detector_backend="skip",
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            emotion = result["dominant_emotion"]
            confidence = result["emotion"][emotion]
            label_text = f"{emotion} {confidence:.0f}%"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    frame_count += 1
    cv2.imshow("DeepFace Emotion", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()