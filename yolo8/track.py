from ultralytics import YOLO
import cv2
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

def convert_boxes_to_deep_sort(yolo_boxes):
    deep_sort_detections = []

    for box in yolo_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = box.cls[0].cpu().numpy()

        deep_sort_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, int(cls)))
    
    return deep_sort_detections

def process_video_display(video_path):
    model = YOLO("runs/detect/train/weights/best.pt")

    # DeepSORT 트래커 초기화
    tracker = DeepSort(
        max_age=30,                # 트랙이 유지되는 최대 프레임 수
        n_init=3,                  # 트랙 초기화에 필요한 연속 탐지 수
        nms_max_overlap=1.0,       # NMS 최대 오버랩 임계값
        max_cosine_distance=0.3,   # 재식별을 위한 코사인 거리 임계값(낮을수록 더 엄격)
        nn_budget=100              # 특징 벡터 저장을 위한 예산
    )

    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_delay = 1 / video_fps  
    
    print(f"원본 비디오 FPS: {video_fps}")

    # 색상 맵 생성 (트래킹된 객체마다 다른 색상 지정)
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(100, 3))
    
    try:
        while cap.isOpened():
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame, verbose=False, conf=0.7, iou=0.5)

            if len(results[0].boxes) > 0:
                detections = convert_boxes_to_deep_sort(results[0].boxes)
                tracks = tracker.update_tracks(detections, frame=frame)

                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    # 트랙 정보 가져오기    
                    track_id = int(track.track_id)
                    ltrb = track.to_ltrb()

                    color = COLORS[track_id % len(COLORS)]
                    x1, y1, x2, y2 = map(int, ltrb)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2)

                    text = f"ID: {track_id}"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
            
            processing_time = time.time() - start_time
            current_fps = 1 / processing_time if processing_time > 0 else 0

            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 8)
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            
            cv2.imshow('YOLO + DeepSORT Tracking', frame)

            wait_time = max(1, int((frame_delay - processing_time) * 1000))
            
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
            
    finally:
        cap.release()
        cv2.destroyAllWindows()

video_path = "../resource/youtube4.mp4"
process_video_display(video_path)