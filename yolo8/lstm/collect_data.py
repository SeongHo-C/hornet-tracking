import cv2
import torch
import pickle
import time
import numpy as np
import os
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

class DataCollector: 
    def __init__(self, video_path):
        self.video_path = video_path
        self.video = None
        self.feature_history = defaultdict(lambda: [])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.model = YOLO("../weights/pose.pt")
        self.model.to(self.device)

    def initialize_video(self):
        self.video = cv2.VideoCapture(self.video_path)
        if not self.video.isOpened():
            raise RuntimeError(f"비디오 파일을 열 수 없습니다: {self.video_path}")

        return self.video.isOpened()

    def calculate_features(self, keypoints, bbox):
        head, heart, tail = keypoints[:3]
    
        features = {
            'head_x': float(head[0]),
            'head_y': float(head[1]),
            'heart_x': float(heart[0]),
            'heart_y': float(heart[1]),
            'tail_x': float(tail[0]),
            'tail_y': float(tail[1]),
        
            'head_heart_angle': float(np.degrees(np.arctan2(
                head[1] - heart[1],
                head[0] - heart[0]
            ))),
            'heart_tail_angle': float(np.degrees(np.arctan2(
                heart[1] - tail[1],
                heart[0] - tail[0]
            ))),
        
            'head_heart_dist': float(np.sqrt(
                (head[0] - heart[0])**2 + 
                (head[1] - heart[1])**2
            )),
            'heart_tail_dist': float(np.sqrt(
                (heart[0] - tail[0])**2 + 
                (heart[1] - tail[1])**2
            )),
        
            'bbox_width': float(bbox[2]),
            'bbox_height': float(bbox[3]),
            'bbox_aspect_ratio': float(bbox[2] / bbox[3]),
            'bbox_area': float(bbox[2] * bbox[3]),
        
            'body_length': float(np.sqrt(
                (head[0] - tail[0])**2 + 
                (head[1] - tail[1])**2
            )),
            'body_orientation': float(np.degrees(np.arctan2(
                tail[1] - head[1],
                tail[0] - head[0]
            ))),
        
            'timestamp': time.time()
        }
    
        return features

    def collect_data(self): 
        if not self.initialize_video():
            print("비디오 초기화 실패")
            return

        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        prev_features = {}

        try:
            while True:
                ret, frame = self.video.read()
                if not ret:
                    break

                with torch.no_grad():
                    results = self.model.track(
                        source=frame,
                        tracker="botsort.yaml",
                        conf=0.3,
                        iou=0.3,
                        persist=True,
                    )

                result = results[0]

                if result.boxes.id is not None and result.keypoints is not None:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()
                    keypoints = result.keypoints.data.cpu().numpy()
                    confidences = result.boxes.conf.cpu()

                    if len(track_ids) > 0:
                        max_conf_idx = confidences.argmax()
                        box = boxes[max_conf_idx]
                        kpts = keypoints[max_conf_idx]

                        fixed_track_id = 0

                        features = self.calculate_features(kpts, box)

                        if fixed_track_id in prev_features:
                            prev = prev_features[fixed_track_id]
                            time_diff = features['timestamp'] - prev['timestamp']

                            features.update({
                                'velocity_x': (features['heart_x'] - prev['heart_x']) / time_diff,
                                'velocity_y': (features['heart_y'] - prev['heart_y']) / time_diff,
                                'angular_velocity': (features['body_orientation'] - prev['body_orientation']) / time_diff,
                                'size_change_rate': (features['body_length'] - prev['body_length']) / time_diff
                            })

                        prev_features[fixed_track_id] = features
                        self.feature_history[fixed_track_id].append(features)

                processed_frames += 1
                progress = (processed_frames / total_frames) * 100
                print(f"\r처리 진행 중... {processed_frames}/{total_frames} 프레임 "
                    f"({progress:.1f}%) 처리됨, 수집된 객체 수: {len(self.feature_history)}", 
                    end='')

                annotated_frame = result.plot()
                cv2.imshow('Data Collection', annotated_frame)
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("\n사용자에 의해 중단되었습니다.")
        finally:
            self.save_data()
            self.cleanup()

    def save_data(self):
        os.makedirs('collected_data', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'collected_data/features_{timestamp}.pkl'
        
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.feature_history), f)

        tracked_frames = len(self.feature_history[0]) if 0 in self.feature_history else 0
        
        print(f"\n데이터가 저장되었습니다: {filename}")
        print(f"수집된 총 객체 수: {len(self.feature_history)}")
        print(f"추적된 프레임 수: {tracked_frames}")

    def cleanup(self):
        if self.video is not None:
            self.video.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "../../resource/giant.mp4"  # 실제 비디오 파일 경로로 변경
    collector = DataCollector(video_path)
    collector.collect_data()


