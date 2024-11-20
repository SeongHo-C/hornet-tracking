from ultralytics import YOLO

model = YOLO("yolov8s.pt") 
results = model("075258.jpg", show=True, save=True)