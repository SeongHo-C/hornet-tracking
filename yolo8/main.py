import asyncio
import websockets
import json
import cv2
import base64 # 바이너리 데이터를 텍스트로 변환하기 위한 라이브러리
import torch
from datetime import datetime
from ultralytics import YOLO
from tracker import ObjectTracker

class VideoProcessor:
    def __init__(self):
        self.camera = None
        self.is_streaming = False
        self.is_detecting = False

        # CUDA(GPU) 사용 가능 여부 확인
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # YOLO 모델 로드 및 GPU 설정
        self.yolo_model = YOLO("weights/second_train.pt")
        self.yolo_model.to(self.device)

        self.tracker = ObjectTracker()

    def initialize_camera(self):
        try:
            if self.camera is None:
                self.camera = cv2.VideoCapture('../resource/youtube5.mp4')
                if not self.camera.isOpened():
                    raise RuntimeError("카메라를 열 수 없습니다.")
            
                # 카메라 설정 최적화
                # self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                # self.camera.set(cv2.CAP_PROP_FPS, 30)
                # self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        except Exception as e:
            print(f"Camera initialization error: {e}")
            self.release_camera()
            raise

    def release_camera(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def process_frame(self, frame):
        with torch.no_grad(): # 추론시 메모리 사용 감소
            results = self.yolo_model(frame, conf = 0.7, iou = 0.5)
            processed_frame = self.tracker.track_objects(frame, results[0].boxes)
        
        return processed_frame
        # return results[0].plot

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
                    new_width = data.get['width']
                    new_height = data.get['height']
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