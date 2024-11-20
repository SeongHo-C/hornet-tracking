from ultralytics import YOLO
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

model = YOLO("yolov8s.pt")

results = model.train(
    data='../datasets/data.yaml',
    epochs=1000,
    batch=16,
    imgsz=640,
    device='0',
    
    # 기본 증강 설정
    fliplr=0.5,          # 좌우반전 50% 확률
    scale=0.5,           # 스케일 변경
    translate=0.1,       # 이미지 이동
    mosaic=0.0,          # 모자이크 비활성화 (초기에는 끄는 것 추천)
    
    # 다른 증강은 기본값 0으로 시작
    hsv_h=0.0,          # HSV 색조
    hsv_s=0.0,          # HSV 채도
    hsv_v=0.0,          # HSV 명도
    degrees=0.0,        # 회전
    shear=0.0,          # 전단
    perspective=0.0,    # 원근
    flipud=0.0,         # 상하반전
    mixup=0.0,          # 믹스업

    # save_period=-1
    # patience=100
)