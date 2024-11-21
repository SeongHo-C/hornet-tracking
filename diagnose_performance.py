from ultralytics import YOLO
import cv2
import time
import psutil
import torch
import gc
from pathlib import Path

def diagnose_performance(video_path, model_path):
    print("\n1. 시스템 정보:")
    print(f"CPU 코어 수: {psutil.cpu_count()}")
    print(f"사용 가능한 RAM: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"사용 가능한 VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    else:
        print("GPU를 사용할 수 없습니다.")

    cap = cv2.VideoCapture(video_path)
    print("\n2. 비디오 정보:")
    print(f"해상도: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}")
    print(f"총 프레임 수: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

    print("\n3. 성능 테스트 시작...")
    model = YOLO(model_path)

    initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)

    frame_times = []
    memory_usage = []
    gpu_memory_usage = []

    try:
        for _ in range(100): # 100 프레임 테스트
            ret, frame = cap.read()
            if not ret: 
                break

            start_time = time.time()
            results = model(frame, verbose=False)
            frame_time = time.time() - start_time
            frame_times.append(frame_time)

            memory_usage.append(psutil.Process().memory_info().rss / (1024 * 1024))

            if torch.cuda.is_available():
                gpu_memory_usage.append(torch.cuda.memory_allocated() / (1024 * 1024))

    finally:
        cap.release()

        print("\n4. 성능 분석 결과:")
        avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
        print(f"평균 FPS: {avg_fps:.1f}")
        print(f"프레임당 평균 처리 시간: {(sum(frame_times) / len(frame_times)) * 1000:.1f}ms")
        print(f"최대 처리 시간: {max(frame_times) * 1000:.1f}ms")

        memory_increase = max(memory_usage) - initial_memory
        print(f"메모리 사용량 증가: {memory_increase:.1f}MB")

        if gpu_memory_usage:
            print(f"최대 GPU 메모리 사용량: {max(gpu_memory_usage):.1f}MB")

        print("\n5. 성능 개선 제안:")
        if avg_fps < 15:
            print("- 프레임 스킵 적용 권장 (skip_frames=2 또는 3)")
        if max(frame_times) * 1000 > 100:
            print("- 입력 해상도 축소 권장 (640x480)")
        if memory_increase > 1000:
            print("- 메모리 최적화 필요")
        if torch.cuda.is_available() and max(gpu_memory_usage) > 3000:
            print("- GPU 메모리 정리 권장")

video_path = "resource/youtube4.mp4"
model_path = "yolo8/runs/detect/train/weights/best.pt"
diagnose_performance(video_path, model_path)