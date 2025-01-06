from ultralytics import YOLO
import cv2
import os
import torch

def process_images_in_folder(input_folder, output_folder):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = YOLO("../weights/detect.pt")
    model.to(device)
    
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            
            results = model(image)
            annotated_image = results[0].plot()
            
            output_path = os.path.join(output_folder, f'detected_{filename}')
            cv2.imwrite(output_path, annotated_image)

# 폴더 경로 설정
input_folder = "../../resource/test_images"  
output_folder = "../../resource/detected_images" 
process_images_in_folder(input_folder, output_folder)