import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class HornetLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2):
        super(HornetLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def prepare_data(sequence_length=10):
    # CSV 파일에서 데이터 로드
    df = pd.read_csv('hornet_coordinates.csv')
    coordinates = df[['x', 'y']].values
    
    # 데이터 정규화
    scaler = MinMaxScaler()
    coordinates_scaled = scaler.fit_transform(coordinates)
    
    # 시퀀스 데이터 생성
    X, y = [], []
    for i in range(len(coordinates_scaled) - sequence_length):
        X.append(coordinates_scaled[i:i + sequence_length])
        y.append(coordinates_scaled[i + sequence_length])
    
    return np.array(X), np.array(y), scaler

def train_model(X, y, epochs=100):
    # 데이터를 PyTorch 텐서로 변환
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    # 모델 초기화
    model = HornetLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # 모델 학습
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

# 데이터 준비 및 모델 학습
X, y, scaler = prepare_data()
model = train_model(X, y)

# 모델 저장
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler
}, 'hornet_lstm_model.pth')

print("Model training completed and saved to hornet_lstm_model.pth")