from ultralytics import YOLO
import time
import cv2
import numpy as np

def test_yolo_streaming(model_path, video_url, display_time=30):
    model = YOLO(model_path)

    # frame_count = 0
    fps_list = []  
    start_time = time.time()

    print("Starting performance test...")

    try:
        results = model.track(
            source=video_url,
            stream=True,
            show=True
        )

        for result in results:
            current_time = time.time()
            if current_time - start_time > display_time:
                break

            frame_time = result.speed['inference'] / 1000.0 
            current_fps = 1 / frame_time if frame_time > 0 else 0
            fps_list.append(current_fps)

            # frame_count += 1

            # if frame_count % 30 == 0:
            #     avg_fps = np.mean(fps_list[-30:])
            #     print(f"Current FPS: {avg_fps:.2f}")

            if cv2.waitKey(1) & 0xFF == 27:
                break
            
    finally:
        cv2.destroyAllWindows()

    if fps_list:
        avg_fps = np.mean(fps_list)
        min_fps = np.min(fps_list)
        max_fps = np.max(fps_list)
        std_fps = np.std(fps_list)

        print("\nPerformance Results:")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Minimum FPS: {min_fps:.2f}")
        print(f"Maximum FPS: {max_fps:.2f}")
        print(f"FPS Standard Deviation: {std_fps:.2f}")
        # print(f"Total Frames Processed: {frame_count}")
        print(f"Total Time: {time.time() - start_time:.2f} seconds")

    return fps_list

if __name__ == "__main__":
    model_path = "yolo11n.pt"
    video_url = "https://www.youtube.com/watch?v=CtfIZzlR3x0"

    try:
        fps_data = test_yolo_streaming(model_path, video_url, display_time=30)
    except Exception as e:
        print(f"Error occurred: {str(e)}")