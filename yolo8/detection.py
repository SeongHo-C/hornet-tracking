from ultralytics import YOLO
import cv2
import time

def process_video_display(video_path):
    model = YOLO("runs/detect/train/weights/best.pt")
    cap = cv2.VideoCapture(video_path)
    
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_delay = 1 / video_fps  
    
    print(f"원본 비디오 FPS: {video_fps}")
    
    try:
        while cap.isOpened():
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()
            
            processing_time = time.time() - start_time
            
            current_fps = 1 / processing_time if processing_time > 0 else 0

            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 8)
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            
            cv2.imshow('YOLO Detection', annotated_frame)
            
            wait_time = max(1, int((frame_delay - processing_time) * 1000))
            
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
            
    finally:
        cap.release()
        cv2.destroyAllWindows()

video_path = "../resource/youtube4.mp4"
process_video_display(video_path)