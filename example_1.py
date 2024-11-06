# Persisting Tracks Loop
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# video_path = "path/to/video.mp4"
cap = cv2.VideoCapture(1)

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