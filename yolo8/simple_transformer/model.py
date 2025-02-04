import os
import time
import math
import torch
import matplotlib.pyplot as plt
import torch.nn as nn 
import numpy as np
import pandas as pd   
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

input_window = 200
output_window = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

module_path = os.path.dirname(__file__)
df = pd.read_csv(module_path + '/synthetic_hornet_sequences2.csv')

# tracking_id별로 데이터 분리해서 시각화
unique_ids = df['tracking_id'].unique()

plt.figure(figsize=(10, 8))
colors = ['blue', 'red']

for idx, track_id in enumerate(unique_ids):
    mask = df['tracking_id'] == track_id
    track_data = df[mask]

    plt.plot(track_data['x'], track_data['y'], color=colors[idx], alpha=0.6, label=f'Track ID {track_id}')

plt.title('Hornet Movement Paths')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 시퀀스의 위치 정보를 인코딩, 말벌의 움직임에서 시간적 순서 정보를 모델에 제공 
class PositionalEncoding(nn.Module):
    # d_model: 좌표의 차원(x, y), max_len: 최대 시퀀스 길이(20) 
    def __init__(self, d_model=2, max_len=200):
       super(PositionalEncoding, self).__init__()
       position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)                  
       pe = torch.zeros(max_len, d_model)                                                     
       pe[:, 0] = torch.sin(position.squeeze() / (10000 ** (0/d_model)))
       pe[:, 1] = torch.cos(position.squeeze() / (10000 ** (1/d_model)))
       pe = pe.unsqueeze(0)
       self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 트랜스포머 기반 시계열 학습 모델 정의
class TransAm(nn.Module):
    def __init__(self, d_model, num_layers, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.pos_encoder = PositionalEncoding(d_model=d_model)

        # 트랜스포머 인코더 레이어
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, # 입력 차원 (x, y 좌표)
            nhead=2,         # 2차원 데이터에 맞춰 헤드 수 조정
            dropout=dropout,
            batch_first=True # 배치 차원을 첫번째로
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )

        # 디코더를 2차원 출력으로 변경 (x, y 좌표 예측)
        self.decoder = nn.Linear(input_window * d_model, output_window * d_model)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.pos_encoder(src)            # 위치 정보 추가
        output = self.transformer_encoder(src) # 트랜스포머 인코더
        output = output.reshape(output.shape[0], -1)
        output = self.decoder(output).reshape(-1, output_window, 2)
        return output

# 학습될 데이터 생성
def create_inout_sequences(input_data):
    sequences = []
    inputs = []
    targets = []
    L = len(input_data)
   
    for i in range(L - input_window - output_window + 1):
        seq = torch.FloatTensor(input_data[i:i + input_window])
        target = torch.FloatTensor(input_data[i + input_window:i + input_window + output_window])
        inputs.append(seq)
        targets.append(target)
   
    inputs = torch.stack(inputs)   # [N, 20, 2]
    targets = torch.stack(targets) # [N, 5, 2]
   
    return inputs, targets

def get_data(df, train_split):
    track_data = df[df['tracking_id'] == 1][['x', 'y']].values
   
    # 원본 데이터 범위 출력
    print("\nOriginal coordinate ranges:")
    print(f"X: {track_data[:, 0].min():.2f} to {track_data[:, 0].max():.2f}")
    print(f"Y: {track_data[:, 1].min():.2f} to {track_data[:, 1].max():.2f}")

    # 해상도 기반 정규화
    normalized_data = track_data.copy()
    normalized_data[:, 0] = normalized_data[:, 0] / 1920  # X 좌표
    normalized_data[:, 1] = normalized_data[:, 1] / 1080  # Y 좌표

    # 정규화된 데이터 범위 출력
    print("\nNormalized coordinate ranges:")
    print(f"X: {normalized_data[:, 0].min():.4f} to {normalized_data[:, 0].max():.4f}")
    print(f"Y: {normalized_data[:, 1].min():.4f} to {normalized_data[:, 1].max():.4f}")
   
    split_idx = int(len(normalized_data) * train_split)
    train_data = normalized_data[:split_idx]
    test_data = normalized_data[split_idx:]
   
    train_inputs, train_targets = create_inout_sequences(train_data)
    test_inputs, test_targets = create_inout_sequences(test_data)
   
    return (train_inputs.to(device), train_targets.to(device)), (test_inputs.to(device), test_targets.to(device))

# 학습될 배치 데이터를 리턴하는 함수를 정의
def get_batch(train_data, i, batch_size):
    seq_len = min(batch_size, len(train_data[0]) - i)
    return train_data[0][i:i + seq_len], train_data[1][i:i + seq_len]

def train(train_data, model, optimizer, criterion, scheduler, epoch, batch_size):
    model.train() # 모델을 학습 모드로 설정
    total_loss = 0.
    total_x_loss = 0.
    total_y_loss = 0.
    start_time = time.time()
    n_batches = 0

    # 배치 단위로 학습 데이터 처리
    for batch, i in enumerate(range(0, len(train_data[0]), batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()

        output = model(data) # 모델의 예측값 계산
        loss = criterion(output, targets) # MSE 손실 계산 (x, y 좌표의 평균 제곱 오차)
        
        # x, y 좌표별 손실 계산
        x_loss = criterion(output[..., 0], targets[..., 0])
        y_loss = criterion(output[..., 1], targets[..., 1])

        loss.backward() # 역전파
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7) # 그래디언트 클리핑 (폭발 방지)
        optimizer.step() # 옵티마이저 스텝

        total_loss += loss.item() # 현재 배치의 손실값을 누적
        total_x_loss += x_loss.item()
        total_y_loss += y_loss.item()
        n_batches += 1 

    # 에폭이 끝난 후 평균 손실 계산
    avg_loss = total_loss / n_batches
    avg_x_loss = total_x_loss / n_batches
    avg_y_loss = total_y_loss / n_batches

    elapsed = time.time() - start_time

    # MSE이므로 제곱근을 취해서 RMSE를 구함
    rmse = math.sqrt(avg_loss)
    rmse_x = math.sqrt(avg_x_loss)
    rmse_y = math.sqrt(avg_y_loss)

    print('| epoch {:3d} | time: {:5.2f}s | loss {:.4f} | rmse_x: {:.4f} | rmse_y: {:.4f} |'.format(
           epoch,
           elapsed,
           avg_loss,  # MSE
           rmse_x,    # RMSE for x (정규화된 상태)
           rmse_y     # RMSE for y (정규화된 상태)
        ))

    return avg_loss

# 평가 함수 정의
def evaluate(eval_model, val_data, criterion):
    eval_model.eval()  # 평가 모드로 설정
    total_loss = 0.
    total_x_loss = 0.
    total_y_loss = 0.
    n_samples = 0
    eval_batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(val_data[0]), eval_batch_size):
            data, targets = get_batch(val_data, i, eval_batch_size)
            
            output = eval_model(data)
            
            # MSE 손실 계산
            loss = criterion(output, targets)

            # x, y 좌표별 손실 계산
            x_loss = criterion(output[..., 0], targets[..., 0])
            y_loss = criterion(output[..., 1], targets[..., 1])
           
            # 배치 크기를 곱해서 누적
            total_loss += loss.item() * len(data)
            total_x_loss += x_loss.item() * len(data)
            total_y_loss += y_loss.item() * len(data)

            n_samples += len(data)

    # 평균 손실 계산
    avg_loss = total_loss / n_samples
    avg_x_loss = total_x_loss / n_samples
    avg_y_loss = total_y_loss / n_samples
   
    # RMSE 계산 (정규화된 상태)
    rmse = math.sqrt(avg_loss)
    rmse_x = math.sqrt(avg_x_loss)
    rmse_y = math.sqrt(avg_y_loss)
    
    # print('-' * 89)
    # print(f'| val_loss: {avg_loss:.4f} | rmse: {rmse:.4f} | rmse_x: {rmse_x:.4f} | rmse_y: {rmse_y:.4f} |')
    # print('-' * 89)
   
    return avg_loss

def test_prediction(model_path, input_sequence):
    # 모델 로드
    model = TransAm(2, 3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 입력 데이터 정규화
    normalized_input = input_sequence.copy()
    normalized_input[:, 0] = normalized_input[:, 0] / 1920  # X 좌표
    normalized_input[:, 1] = normalized_input[:, 1] / 1080  # Y 좌표
    
    # 텐서로 변환
    normalized_input = torch.from_numpy(normalized_input).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(normalized_input)
        
        # 예측값 역정규화
        predicted_coords = output.squeeze(0).cpu().numpy()
        predicted_coords[:, 0] = predicted_coords[:, 0] * 1920  # X 좌표
        predicted_coords[:, 1] = predicted_coords[:, 1] * 1080  # Y 좌표

        # 입력 시퀀스와 예측 결과 시각화
        plt.figure(figsize=(12, 8))
        # x축 범위 설정 (시작값, 끝값)
        plt.xlim(0, 800)

        # y축 범위 설정 (시작값, 끝값)
        plt.ylim(0, 600)
        
        # 입력 시퀀스 (파란색)
        plt.plot(input_sequence[:, 0], input_sequence[:, 1], 'b-', label='Input sequence', marker='o')
        
        # 예측 시퀀스 (빨간색)
        for i, (x, y) in enumerate(predicted_coords):
            plt.plot(x, y, 'r--') 
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')  # 숫자 표시
        
        print(predicted_coords)

        plt.title('Hornet Movement Prediction')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True)
        plt.legend()
        plt.show()

        # 예측된 좌표값 출력
        print("\nPredicted next 5 positions:")
        for i, (x, y) in enumerate(predicted_coords, 1):
            print(f"Position {i}: ({x:.2f}, {y:.2f})")

        return predicted_coords   

# # 데이터 분할
# train_data, val_data = get_data(df, 0.8) 

# # 모델 초기화
# model = TransAm(2, 3).to(device)

# # 손실 함수와 옵티마이저 설정
# criterion = nn.MSELoss()
# lr = 0.0001 
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# # 학습 설정
# epochs = 100
# batch_size = 32 # 신경망 학습 과정에서 한 번에 처리하는 데이터 샘플의 개수
# best_val_loss = float('inf')

# # 학습 루프
# for epoch in range(1, epochs + 1):
#     epoch_start_time = time.time()
    
#     # 학습
#     train_loss = train(train_data, model, optimizer, criterion, scheduler, epoch, batch_size)
    
#     # 검증
#     val_loss = evaluate(model, val_data, criterion)

#     scheduler.step()

#     # 최적 모델 저장
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), 'best_hornet_model2.pth')
    
# 예측 테스트
test_idx = 400
test_sequence = df[df['tracking_id'] == 1][['x', 'y']].values[test_idx:test_idx+200]
predicted_positions = test_prediction('best_hornet_model2.pth', test_sequence)