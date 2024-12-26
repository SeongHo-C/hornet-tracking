from ultralytics import YOLO
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

model = YOLO("yolov8s.pt")

results = model.train(
    data = '../datasets/data.yaml',
    device = '0',

    # 학습 기본 설정
    epochs = 300, 
    patience = 30,      # 조기종료 인내심 
    warmup_epochs = 10, # 학습 초기에 학습률을 점진적으로 증가시키는 기간
    imgsz = 800,
    batch = 24,

    # 옵티마이저 설정
    lr0 = 0.001,           # 초기 학습률            
    lrf = 0.0001,          # 최종 학습률 => 과적합 방지, 정확도 향상
    weight_decay = 0.0005, # 가중치 감쇠, 과적합 방지
    optimizer = 'AdamW',   # momentum과 적응적 학습률을 결합한 최적화 알고리즘
    
    # 데이터 증강
    fliplr = 0.5,    # 좌우반전 50% 확률 
    scale = 0.3,     # 스케일 변경 ±40%        
    translate = 0.1, # 이동 ±10%    
    mosaic = 0.8,    # 모자이크 증강 80% 확률
    hsv_h = 0.015,   # 색조 ±1.5% 
    hsv_s = 0.7,     # 채도 ±70%
    hsv_v = 0.4,     # 명도 ±40%
    degrees = 10,    # 회전 ±10도
    
    # 추가 설정
    save_period = -1,
    box = 4.5,    # 바운딩 박스 loss 가중치
    cls = 1.0,    # 분류 loss 가중치
    plots = True, # 학습 그래프 저장
)