import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

class HornetSequenceCollector:
    def __init__(self, model_path, video_path, output_file, max_time_gap):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.output_file = output_file
        self.max_time_gap = max_time_gap

        self.last_detection_time = None
        self.tracking_id = 1

        # 빈 DataFrame 생성
        self.df = pd.DataFrame(columns=['tracking_id', 'datetime', 'x', 'y'])
        # CSV 파일 생성
        self.df.to_csv(output_file, index=False)

    def get_coordinates(self, thorax, head, abdomen):
        # 좌표가 0인 경우는 None 처리
        if thorax is not None and thorax[0] == 0 and thorax[1] == 0:
            thorax = None
        if head is not None and head[0] == 0 and head[1] == 0:
            head = None
        if abdomen is not None and abdomen[0] == 0 and abdomen[1] == 0:
            abdomen = None

        if thorax is not None:
            return thorax[0], thorax[1]
        elif head is not None:
            return head[0], head[1]
        elif abdomen is not None:
            return abdomen[0], abdomen[1]
        
        return None, None

    def record_detection(self, thorax, head, abdomen):
        current_time = time.time()
        x, y = self.get_coordinates(thorax, head, abdomen)

        if x is None:
            return False

        # 연속성 체크
        if self.last_detection_time is None:
            is_continuous = True
        else:
            time_gap = current_time - self.last_detection_time
            is_continuous = time_gap <= self.max_time_gap
            if not is_continuous:
                self.tracking_id += 1

        self.last_detection_time = current_time

        new_data = {
            'tracking_id': self.tracking_id,
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            'x': x,
            'y': y
        }

        # DataFrame에 추가하고 CSV 파일에 저장
        pd.DataFrame([new_data]).to_csv(self.output_file, mode='a', header=False, index=False)

        return is_continuous

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = self.model(frame, verbose=False, conf=0.3, iou=0.3)
            boxes = results[0].boxes
            keypoints = results[0].keypoints

            if len(boxes) > 0 and len(keypoints) > 0:
                max_conf_idx = 0
                max_conf = 0

                for idx, box in enumerate(boxes):
                    if float(box.conf) > max_conf:
                        max_conf = float(box.conf)
                        max_conf_idx = idx

                kpts = keypoints[max_conf_idx]
                kpts_data = kpts.data[0]

                # 각 키포인트 추출
                thorax = [float(kpts_data[1][0]), float(kpts_data[1][1]), float(kpts_data[1][2])] if kpts_data[1][2] >= 0.3 else None
                head = [float(kpts_data[0][0]), float(kpts_data[0][1]), float(kpts_data[0][2])] if kpts_data[0][2] >= 0.3 else None
                abdomen = [float(kpts_data[2][0]), float(kpts_data[2][1]), float(kpts_data[2][2])] if kpts_data[2][2] >= 0.3 else None
                       
                # 위치 기록
                self.record_detection(thorax, head, abdomen)

                annotated_frame = results[0].plot()

                if thorax is not None:
                    cv2.putText(annotated_frame, f'ID: {self.tracking_id}', (int(thorax[0]), int(thorax[1]) - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                    cv2.imshow('Hornet Detection', annotated_frame)
                        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()              

        print(f'Video processing completed. Results saved to {self.output_file}')  

def main():
    collector = HornetSequenceCollector(
        model_path = '../weights/pose.pt',
        video_path = '../../resource/giant.mp4',
        output_file = 'hornet_sequences.csv',
        max_time_gap = 0.5
    )

    collector.process_video()
     
if __name__ == '__main__':
    main()
