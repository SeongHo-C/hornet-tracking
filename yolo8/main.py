import asyncio
import websockets
import json
import cv2
import base64 # 바이너리 데이터를 텍스트로 변환하기 위한 라이브러리
import torch
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict
from transformer.model import TransAm

class VideoProcessor:
    def __init__(self):
        self.camera = None
        self.is_streaming = False
        self.is_detecting = False

        # CUDA(GPU) 사용 가능 여부 확인
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {self.device}')
        
        # YOLO 모델 로드 및 GPU 설정
        self.yolo_model = YOLO("weights/pose.pt")
        self.yolo_model.to(self.device)

        # Transformer 모델 로드
        self.trans_model = TransAm(2, 3).to(self.device)
        model_path = os.path.join(os.path.dirname(__file__), 'transformer/hornet_model.pth')
        self.trans_model.load_state_dict(torch.load(model_path))
        self.trans_model.eval()

        self.track_history = defaultdict(lambda: [])
        self.prediction_history = {}

    def initialize_camera(self):
        try:
            if self.camera is None:
                self.camera = cv2.VideoCapture('../resource/giant.mp4')
                if not self.camera.isOpened():
                    raise RuntimeError("카메라를 열 수 없습니다.")

                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            
        except Exception as e:
            print(f'Camera initialization error: {e}')
            self.release_camera()
            raise

    def release_camera(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def predict_future_positions(self, track_history):
        if len(track_history) < 25:  # 최소 25개의 포인트가 필요
            return None
            
        # 최근 25개 포인트 사용
        recent_points = np.array(track_history[-25:])
        
        # 입력 정규화
        normalized_input = recent_points.copy()
        normalized_input[:, 0] = normalized_input[:, 0] / 800
        normalized_input[:, 1] = normalized_input[:, 1] / 600
        
        # 텐서로 변환
        normalized_input = torch.from_numpy(normalized_input).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.trans_model(normalized_input)
            
            # 예측값 역정규화
            predicted_coords = output.squeeze(0).cpu().numpy()
            predicted_coords[:, 0] = predicted_coords[:, 0] * 800
            predicted_coords[:, 1] = predicted_coords[:, 1] * 600
            
            return predicted_coords

    def process_frame(self, frame):
        with torch.no_grad(): # 추론시 메모리 사용 감소
            results = self.yolo_model.track(
                source=frame, 
                tracker="botsort.yaml", 
                conf=0.5, 
                iou=0.5, 
                persist=True
            )

            result = results[0]
            annotated_frame = result.plot()

            if result.boxes.id is None:
                return annotated_frame
            
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = self.track_history[0]
                track.append((float(x), float(y)))

                if len(track) > 30:
                    track = track[-30:]

                self.track_history[0] = track

                if len(track) > 1:
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                # 예측 수행 및 시각화
                if len(track) >= 25:
                    predicted_positions = self.predict_future_positions(track)
                    if predicted_positions is not None:
                        # 예측 경로 시각화 (빨간색)
                        pred_points = np.array(predicted_positions, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [pred_points], isClosed=False, color=(0, 0, 255), thickness=3)
                        
                        # 예측 포인트 표시
                        for i, (pred_x, pred_y) in enumerate(predicted_positions):
                            cv2.circle(annotated_frame, (int(pred_x), int(pred_y)), 
                                     radius=3, color=(0, 0, 255), thickness=-1)
                            cv2.putText(annotated_frame, str(i), (int(pred_x), int(pred_y)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        return annotated_frame

    def change_resolution(self, width, height):
        if self.camera is not None and self.camera.isOpened():
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"Resolution changed to {width}x{height}")
        else:
            print("Camera is not initialized")

    def get_frame(self):
        if self.camera is None or not self.camera.isOpened():
            return None

        ret, frame = self.camera.read()                                
        if not ret:
            return None

        # frame = cv2.resize(frame, (640, 480)) # 프레임 크기 축소

        processed_frame = self.process_frame(frame) if self.is_detecting else frame
        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]) # 화질 80%로 조정

        # base64는 바이너리 데이터(16진수)를 텍스트로 안전하게 전송하기 위한 인코딩 방식
        # 웹소켓은 텍스트 기반 프로토콜이라 바이너리 이미지를 직접 전송할 수 없음
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_text

    async def stream_camera(self, websocket):
        while self.is_streaming:
            frame = self.get_frame()
            if frame is not None:
                try:
                    # 프레임을 JSON 형태로 웹소켓을 통해 전송
                    await websocket.send(json.dumps({
                        'type': 'frame',
                        'data': frame,
                        'timestamp': datetime.now().isoformat()
                    }))
                except websockets.exceptions.ConnectionClosed:
                    break
            # 초당 프레임 수(FPS)를 약 30으로 제한
            await asyncio.sleep(0.033)

    async def handle_message(self, websocket, message):
        try:
            # 클라이언트로부터 받은 JSON 메시지 파싱
            data = json.loads(message)
            command_type = data.get('type')
            command = data.get('action')

            if command_type == 'camera':
                if command == 'start':
                    if not self.is_streaming:
                        print("Starting camera stream")
                        self.initialize_camera()
                        self.is_streaming = True
                        # 비동기 함수를 백그라운드에서 실행하여 다른 작업들과 동시에 처리할 수 있게 해주는 기능
                        asyncio.create_task(self.stream_camera(websocket))
                        await websocket.send(json.dumps({
                            'type': 'response',
                            'message': 'Camera streaming started'
                        }))
                elif command == 'stop':
                    print("Stopping camera stream")
                    self.is_streaming = False
                    self.release_camera()
                    await websocket.send(json.dumps({
                        'type': 'response',
                        'message': 'Camera streaming stopped'
                    }))
                elif command == 'resolution_change':
                    data = data.get('data')
                    new_width = data.get('width')
                    new_height = data.get('height')

                    if new_width and new_height:
                        self.change_resolution(new_width, new_height)
                        await websocket.send(json.dumps({
                            'type': 'response',
                            'message': f'Resolution changed to {new_width}x{new_height}'
                        }))
            elif command_type == 'detection':
                if command == 'start':
                    if not self.is_detecting:
                        print("Starting detection model")
                        self.is_detecting = True
                        await websocket.send(json.dumps({
                            'type': 'response',
                            'message': 'Detection model started'
                        }))
                elif command == 'stop':
                    print("Stopping detection model")
                    self.is_detecting = False
                    await websocket.send(json.dumps({
                        'type': 'response',
                        'message': 'Detection model stopped'
                    }))
            else:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': f'Unknown command: {command}'
                }))

        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Invalid JSON format'
            }))

    async def handle_connection(self, websocket, path=None):
        print("New connection established")
        try:
            await websocket.send(json.dumps({
                'type': 'info',
                'message': 'Connected to Video Processor'
            }))

            # 웹소켓으로부터 메시지가 들어올 때마다 반복해서 처리하는 비동기 반복문
            async for message in websocket:
                await self.handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")

        finally:
            self.is_streaming = False
            self.release_camera()
            cv2.destroyAllWindows()

async def main():
    processor = VideoProcessor()

    # 웹소켓 서버 시작
    # async with: 비동기 컨텍스트 매니저로, 서버의 리소스를 자동으로 관리
    # processor.handle_connection: 새로운 클라이언트 연결이 들어올 때마다 실행될 핸들러 함수
    async with websockets.serve(processor.handle_connection, "localhost", 8765, ping_interval=None):
        print("Video Processor started on ws://localhost:8765")
        # 서버를 계속 실행 상태로 유지, 이게 없으면 서버가 바로 종료
        await asyncio.Future()

if __name__ == "__main__":
    # 비동기 메인 함수 실행
    asyncio.run(main())