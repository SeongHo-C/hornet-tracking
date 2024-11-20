from ultralytics import YOLO

model = YOLO("hornet_25971.pt") 
results = model.track("090223.jpg", show=True, save=True) 