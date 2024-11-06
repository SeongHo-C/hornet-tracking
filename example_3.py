# Plotting Tracks Over Time
from collections import defaultdict
from ultralytics import YOLO

import cv2
import numpy as np

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(1)

track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, conf=0.7, persist=True)

        if results and len(results) > 0:
            boxes = results[0].boxes.xywh.cpu()
            if len(boxes) > 0:
                for box in boxes:
                    x, y, w, h = box

                    print(f"Wasp detected at: {x}, {y}")

        annotated_frame = results[0].plot()
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()