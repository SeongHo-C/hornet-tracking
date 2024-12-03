import cv2
import os
import numpy as np
import pandas as pd
from collections import deque
from ultralytics import YOLO
from tqdm import tqdm

class ObjectTracker:
    def __init__(self, model_path, time_window, batch_size):
        self.time_window = time_window
        self.points = deque(maxlen=time_window)
        self.tracking_data = []
        self.batch_size = batch_size
        
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise Exception(f"모델 로드 중 에러 발생: {e}")
        
    # 객체의 속도를 계산
    def _calculate_velocity(self, current_pos, prev_pos, frame_width, frame_height, time_diff):
        if prev_pos is None:
            return 0.0, 0.0
            
        # 프레임 크기로 나누는 이유는 속도를 정규화하기 위함 (0~1 사이 값으로)    
        velocity_x = (current_pos[0] - prev_pos[0]) / (frame_width * time_diff)
        velocity_y = (current_pos[1] - prev_pos[1]) / (frame_height * time_diff)

        return velocity_x, velocity_y

    def _save_batch(self, output_path, is_final=False):
        if not self.tracking_data:
            return

        df = pd.DataFrame(self.tracking_data)

        # 결측치가 있는 행 제거
        df = df.dropna()
        
        # 신뢰도 기반 필터링
        df = df[df['confidence'] > 0.7]
        
        mode = 'w' if is_final or not os.path.exists(output_path) else 'a'
        header = not os.path.exists(output_path) or is_final
    
        df.to_csv(output_path, mode=mode, header=header, index=False)
    
        if not is_final:
            self.tracking_data = []

    def collect_data(self, video_path, output_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")

        cap = cv2.VideoCapture(video_path)

        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        
        try:
            pbar = tqdm(total=total_frames, desc="Processing frames")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                current_time = frame_count / video_fps
                
                results = self.model(frame, verbose=False, conf=0.7, iou=0.5)
                annotated_frame = results[0].plot()

                if len(results[0].boxes) > 0:
                    box = results[0].boxes[0]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    norm_x = center_x / frame_width
                    norm_y = center_y / frame_height

                    prev_point = self.points[-1] if len(self.points) > 0 else None
                    velocity_x, velocity_y = self._calculate_velocity(
                        (center_x, center_y),
                        prev_point,
                        frame_width,
                        frame_height,
                        1/video_fps
                    )

                    self.tracking_data.append({
                        'frame': frame_count,
                        'timestamp': current_time,
                        'x': center_x,
                        'y': center_y,
                        'norm_x': norm_x,
                        'norm_y': norm_y,
                        'velocity_x': velocity_x,
                        'velocity_y': velocity_y,
                        'confidence': float(box.conf[0]),
                        'width': (x2 - x1) / frame_width,
                        'height': (y2 - y1) / frame_height
                    })

                    self.points.append((center_x, center_y))
                    if len(self.points) >= 2:
                        points_list = list(self.points)
                        for i in range(len(points_list) - 1):
                            cv2.line(annotated_frame, points_list[i], points_list[i + 1], (0, 255, 0), 2)

                    cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(annotated_frame, f"X: {center_x}, Y: {center_y}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('YOLO Detection', annotated_frame)

                if len(self.tracking_data) >= self.batch_size:
                    self._save_batch(output_path)

                pbar.update(1)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"처리 중 에러 발생: {e}")
            
        finally:
            self._save_batch(output_path, is_final=True)
            
            print("\nData Statistics:")
            print(f"Total frames processed: {frame_count}")
            print(f"Total detections: {len(self.tracking_data)}")
            if frame_count > 0:
                print(f"Detection rate: {len(self.tracking_data)/frame_count*100:.2f}%")

            cap.release()
            cv2.destroyAllWindows()
            pbar.close()

if __name__ == "__main__":
    video_path = "../resource/youtube5-1.mp4"
    output_path = "hornet_coordinates.csv"
    model_path = "800-24-epoch163.pt"
    time_window = 30
    batch_size = 500
    
    try:
        tracker = ObjectTracker(model_path, time_window, batch_size)
        tracker.collect_data(video_path, output_path)
    except Exception as e:
        print(f"Error: {e}")