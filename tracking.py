from ultralytics import YOLO

model = YOLO("yolo11n.pt") # Load an official Detect model
# model = YOLO("yolo11n-pose.pt") # Load an official Pose model

results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.80, show=True) # Tracking with default tracker
# results = model.track("https://www.youtube.com/watch?v=1uaZRVTbaws", show=True, tracker="bytetrack.yaml") # with ByteTrack