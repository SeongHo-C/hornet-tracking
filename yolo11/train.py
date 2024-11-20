from ultralytics import YOLO

model = YOLO("hornet_25971.pt")

results = model.train(
    data="./datasets/data.yaml", 
    epochs=24029,
    patience=0,

    # 학습률 (미세조정)
    lr0=0.0001,
    lrf=0.00001,

    # 강화된 데이터 증강
    degrees=30.0,
    translate=0.3,
    scale=1.0,
    shear=5.0,
    perspective=0.001,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.4,

    save_period=10000
)