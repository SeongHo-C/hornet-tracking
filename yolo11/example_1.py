# Persisting Tracks Loop
import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

img_path = "https://www.youtube.com/watch?v=p2UaoHyM8SI"
cap = cv2.VideoCapture(img_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()