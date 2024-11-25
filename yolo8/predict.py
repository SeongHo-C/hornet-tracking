from ultralytics import YOLO
import cv2
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints

class WaspTracker:
    def __init__(self):
        # UKF 초기화
        points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2., kappa=-1)
        self.ukf = UnscentedKalmanFilter(
            dim_x=4,  # [x, y, vx, vy]
            dim_z=2,  # [x, y] 측정
            dt=0.1,
            fx=self.motion_model,
            hx=self.measurement_model,
            points=points
        )

        # 행동 상태 및 파라미터
        self.current_behavior = 'cruising'
        self.behavior_params = {
            'hovering': {
                'Q': np.diag([0.1, 0.1, 0.1, 0.1]),
                'R': np.diag([0.5, 0.5]),
                'dt': 0.2
            },
            'cruising': {
                'Q': np.diag([0.5, 0.5, 0.5, 0.5]),
                'R': np.diag([1.0, 1.0]),
                'dt': 0.3
            },
            'maneuvering': {
                'Q': np.diag([1.0, 1.0, 1.0, 1.0]),
                'R': np.diag([1.5, 1.5]),
                'dt': 0.1
            }
        }

        # 초기 노이즈 설정
        self.ukf.Q = self.behavior_params['cruising']['Q']
        self.ukf.R = self.behavior_params['cruising']['R']
        self.initialized = False
        self.prev_pos = None

        self.prediction_history = {}  # {timestamp: [(predicted_x, predicted_y, step), ...]}
        self.position_history = []    # [(timestamp, x, y), ...]

    def motion_model(self, x, dt):
        # 상태 전이 함수
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return F @ x

    def measurement_model(self, x):
        # 측정 함수 (위치만 관측)
        return x[:2]    

    def update_behavior(self, position):
        if self.prev_pos is None:
            self.prev_pos = position
            return
        
        # 속도 계산
        velocity = (position - self.prev_pos) / 0.1  # dt = 0.1
        speed = np.linalg.norm(velocity)

         # 행동 상태 업데이트
        if speed < 0.5:
            self.current_behavior = 'hovering'
        elif speed > 2.0:
            self.current_behavior = 'maneuvering'
        else:
            self.current_behavior = 'cruising'
        
        self.prev_pos = position
        
        # 현재 행동에 따른 파라미터 적용
        params = self.behavior_params[self.current_behavior]
        self.ukf.Q = params['Q']
        self.ukf.R = params['R']

    def update(self, measurement):
        current_time = time.time()
        self.position_history.append((current_time, measurement[0], measurement[1]))

        # 오래된 기록 삭제 (예: 5초 이상 된 기록)
        old_threshold = current_time - 5.0
        self.position_history = [(t, x, y) for t, x, y in self.position_history if t > old_threshold]

        # 예측 정확도 계산
        self.calculate_prediction_accuracy()

        if not self.initialized:
            self.ukf.x = np.array([measurement[0], measurement[1], 0, 0])
            self.initialized = True
            self.update_behavior(measurement)
            return self.ukf.x, self.current_behavior
        
        self.ukf.predict()
        self.ukf.update(measurement)
        self.update_behavior(measurement)
        
        return self.ukf.x, self.current_behavior
    
    def calculate_prediction_accuracy(self):
        current_time = time.time()
        errors = []
        
        # 현재 시간과 가장 가까운 과거 예측들 검사
        for pred_time, (pred_x, pred_y, step) in list(self.prediction_history.items()):
            if pred_time <= current_time:
                # 가장 가까운 실제 위치 찾기
                closest_actual = min(self.position_history, 
                                  key=lambda x: abs(x[0] - pred_time))
                
                if abs(closest_actual[0] - pred_time) < 0.1:  # 100ms 이내
                    # 예측 오차 계산
                    error_x = abs(pred_x - closest_actual[1])
                    error_y = abs(pred_y - closest_actual[2])
                    error_dist = np.sqrt(error_x**2 + error_y**2)
                    prediction_time = (pred_time - (current_time - step * self.behavior_params[self.current_behavior]['dt']))
                    
                    errors.append((step, prediction_time, error_dist))
                    print(f"Prediction Step {step} ({prediction_time:.2f}s ahead):")
                    print(f"Predicted: ({pred_x:.2f}, {pred_y:.2f})")
                    print(f"Actual: ({closest_actual[1]:.2f}, {closest_actual[2]:.2f})")
                    print(f"Error Distance: {error_dist:.2f} pixels")
                    print("----------------------")
                
                # 사용한 예측 삭제
                del self.prediction_history[pred_time]
    
    def predict_next(self, steps=5):
        if not self.initialized:
            return None
        
        current_time = time.time()
        predictions = []
        state = self.ukf.x.copy()
        dt = self.behavior_params[self.current_behavior]['dt']
        
        for step in range(steps):
            state = self.motion_model(state, dt)
            predictions.append(state[:2])

            # 예측 시간과 위치 저장
            prediction_time = current_time + (step + 1) * dt
            self.prediction_history[prediction_time] = (state[0], state[1], step + 1)
                        
        return np.array(predictions)

            

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

    tracker = DeepSort(
        max_age=30,               
        n_init=3,                 
        nms_max_overlap=1.0,       
        max_cosine_distance=0.3,   
        nn_budget=100              
    )

    wasp_trackers = {}

    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_delay = 1 / video_fps  
    
    print(f"원본 비디오 FPS: {video_fps}")

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

                    track_id = int(track.track_id)
                    ltrb = track.to_ltrb()

                    center_x = (ltrb[0] + ltrb[2]) / 2
                    center_y = (ltrb[1] + ltrb[3]) / 2

                     # WaspTracker 업데이트
                    if track_id not in wasp_trackers:
                        wasp_trackers[track_id] = WaspTracker()
                    
                    state, behavior = wasp_trackers[track_id].update(np.array([center_x, center_y]))
                    predictions = wasp_trackers[track_id].predict_next()

                    color = COLORS[track_id % len(COLORS)]
                    x1, y1, x2, y2 = map(int, ltrb)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2)

                    text = f"ID: {track_id} ({behavior})"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
            
                    # 현재 위치와 예측 경로 표시
                    if predictions is not None:
                        # 현재 위치에 점 표시
                        cv2.circle(frame, (int(center_x), int(center_y)), 4, (0, 0, 255), -1)
        
                        # 예측 경로 표시
                        for i in range(len(predictions)-1):
                            start = predictions[i]
                            end = predictions[i+1]
                            cv2.line(frame, 
                                    (int(start[0]), int(start[1])),
                                    (int(end[0]), int(end[1])),
                                    (0, 255, 0), 2)
                        
                        # 예측 위치에 점 표시
                        cv2.circle(frame, (int(end[0]), int(end[1])), 3, (0, 255, 0), -1)
                            
            processing_time = time.time() - start_time
            current_fps = 1 / processing_time if processing_time > 0 else 0

            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 8)
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            
            cv2.imshow('YOLO + DeepSORT + UKF Tracking', frame)

            wait_time = max(1, int((frame_delay - processing_time) * 1000))
            
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
            
    finally:
        cap.release()
        cv2.destroyAllWindows()

video_path = "../resource/youtube4.mp4"
process_video_display(video_path)

"""
테스트하면서 조정이 필요한 부분

1. 행동 상태 전환 임계값 (speed < 0.5, speed > 2.0)
2. 각 행동별 노이즈 파라미터 (Q, R)
3. 예측 스텝 수 (현재 5단계)
4. 시각화 방식
"""