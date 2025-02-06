import os
import torch
import pandas as pd   
import matplotlib.pyplot as plt
from model import TransAm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

module_path = os.path.dirname(__file__)
df = pd.read_csv(module_path + '/synthetic_hornet_sequences.csv')

def test_prediction(model_path, input_sequence):
    model = TransAm(2, 3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    normalized_input = input_sequence.copy()
    normalized_input[:, 0] = normalized_input[:, 0] / 800 
    normalized_input[:, 1] = normalized_input[:, 1] / 600 
    
    # 텐서로 변환
    normalized_input = torch.from_numpy(normalized_input).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(normalized_input)
        
        # 예측값 역정규화
        predicted_coords = output.squeeze(0).cpu().numpy()
        predicted_coords[:, 0] = predicted_coords[:, 0] * 800 
        predicted_coords[:, 1] = predicted_coords[:, 1] * 600 

        plt.figure(figsize=(12, 8))
        plt.xlim(0, 800)
        plt.ylim(0, 600)
        
        # 입력 시퀀스 (파란색)
        plt.plot(input_sequence[:, 0], input_sequence[:, 1], 'b-', label='Input sequence')
        for i, (x, y) in enumerate(input_sequence):
            plt.scatter(x, y, c='blue', s=50)
        
        # 예측 시퀀스 (빨간색)
        plt.plot(predicted_coords[:, 0], predicted_coords[:, 1], 'r-', label='Predicted sequence')
        for i, (x, y) in enumerate(predicted_coords):
            plt.scatter(x, y, c='red', s=50)
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.title('Hornet Movement Prediction')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True)
        plt.legend()
        plt.show()

        # # 예측된 좌표값 출력
        # print("\nPredicted next 5 positions:")
        # for i, (x, y) in enumerate(predicted_coords, 1):
        #     print(f"Position {i}: ({x:.2f}, {y:.2f})")

        return predicted_coords

# 예측 테스트
test_idx = 400
test_sequence = df[df['tracking_id'] == 1][['x', 'y']].values[test_idx:test_idx+25]
predicted_positions = test_prediction('hornet_model.pth', test_sequence)