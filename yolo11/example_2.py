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
        results = model.track(frame, persist=True)

        # 객체가 감지되지 않은 경우 처리
        if not hasattr(results[0].boxes, 'id') or results[0].boxes.id is None:
            continue
        
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y))) # x, y center point
            if len(track) > 30:
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        cv2.imshow("YOLO11 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()