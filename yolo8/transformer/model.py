import os
import time
import math
import torch
import torch.nn as nn 
import numpy as np
import pandas as pd   

input_window = 25
output_window = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

module_path = os.path.dirname(__file__)
df = pd.read_csv(module_path + '/hornet_sequences.csv')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=25):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        pe = torch.zeros(max_len, d_model)                                                     
    #    pe[:, 0] = torch.sin(position.squeeze() / (10000 ** (0/d_model)))
    #    pe[:, 1] = torch.cos(position.squeeze() / (10000 ** (1/d_model)))

    # 4개 특성(x, y, direction, speed)에 대한 인코딩
        for i in range(d_model):
            pe[:, i] = torch.sin(position.squeeze() / (10000 ** (i/d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransAm(nn.Module):
    def __init__(self, d_model, num_layers, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        # d_model을 4로 변경 (x, y, direction, speed)
        self.pos_encoder = PositionalEncoding(4)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=4, 
            nhead=2,       
            dropout=dropout,
            batch_first=True 
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )

        # 디코더 출력 차원 수정
        self.decoder = nn.Linear(input_window * 4, output_window * 4)

        # 가중치 초기화
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.pos_encoder(src)            
        output = self.transformer_encoder(src)
        output = output.reshape(output.shape[0], -1)
        output = self.decoder(output).reshape(-1, output_window, 4)
        return output

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
   
    inputs = torch.stack(inputs)   
    targets = torch.stack(targets) 
   
    return inputs, targets

def get_data(df, train_split):
    track_data = df[df['tracking_id'] == 1][['x', 'y', 'direction', 'speed']].values
   
    # # 원본 데이터 범위 출력
    # print("\nOriginal coordinate ranges:")
    # print(f"X: {track_data[:, 0].min():.2f} to {track_data[:, 0].max():.2f}")
    # print(f"Y: {track_data[:, 1].min():.2f} to {track_data[:, 1].max():.2f}")

    # 정규화
    normalized_data = track_data.copy()
    normalized_data[:, 0] = normalized_data[:, 0] / 1920
    normalized_data[:, 1] = normalized_data[:, 1] / 1080
    normalized_data[:, 2] = normalized_data[:, 2] / 180
    normalized_data[:, 3] = (normalized_data[:, 3] - df['speed'].min()) / (df['speed'].max() - df['speed'].min())
   
    split_idx = int(len(normalized_data) * train_split)
    train_data = normalized_data[:split_idx]
    test_data = normalized_data[split_idx:]
   
    train_inputs, train_targets = create_inout_sequences(train_data)
    test_inputs, test_targets = create_inout_sequences(test_data)
   
    return (train_inputs.to(device), train_targets.to(device)), (test_inputs.to(device), test_targets.to(device))

def get_batch(train_data, i, batch_size):
    seq_len = min(batch_size, len(train_data[0]) - i)
    return train_data[0][i:i + seq_len], train_data[1][i:i + seq_len]

def train(train_data, model, optimizer, criterion, scheduler, epoch, batch_size):
    model.train() 
    total_loss = 0.
    # total_x_loss = 0.
    # total_y_loss = 0.
    # total_dir_loss = 0.
    # total_speed_loss = 0.
    start_time = time.time()
    n_batches = 0

    # 배치 단위로 학습 데이터 처리
    for batch, i in enumerate(range(0, len(train_data[0]), batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()

        output = model(data) # 모델의 예측값 계산
        loss = criterion(output, targets) # MSE 손실 계산 (x, y 좌표의 평균 제곱 오차)
        
        # x_loss = criterion(output[..., 0], targets[..., 0])
        # y_loss = criterion(output[..., 1], targets[..., 1])
        # dir_loss = criterion(output[..., 2], targets[..., 2])
        # speed_loss = criterion(output[..., 3], targets[..., 3])

        loss.backward() # 역전파
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7) # 그래디언트 클리핑 (폭발 방지)
        optimizer.step() # 옵티마이저 스텝

        total_loss += loss.item()
        # total_x_loss += x_loss.item()
        # total_y_loss += y_loss.item()
        # total_dir_loss += dir_loss.item()
        # total_speed_loss += speed_loss.item()
        n_batches += 1
        
    avg_loss = total_loss / n_batches
    return avg_loss

def evaluate(eval_model, val_data, criterion):
    eval_model.eval() 
    total_loss = 0.
    total_coord_loss = 0.
    total_dir_loss = 0.
    total_speed_loss = 0.
    n_samples = 0
    eval_batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(val_data[0]), eval_batch_size):
            data, targets = get_batch(val_data, i, eval_batch_size)
            
            output = eval_model(data)
            
            # 전체 손실
            loss = criterion(output, targets)
    
            # 개별 특성 손실
            coord_loss = criterion(output[..., :2], targets[..., :2])
            dir_loss = criterion(output[..., 2], targets[..., 2])
            speed_loss = criterion(output[..., 3], targets[..., 3])
           
            total_loss += loss.item() * len(data)
            total_coord_loss += coord_loss.item() * len(data)
            total_dir_loss += dir_loss.item() * len(data)
            total_speed_loss += speed_loss.item() * len(data)

            n_samples += len(data)

            print('-' * 90)
            print(f'Validation Losses:')
            print(f'Total Loss: {total_loss/n_samples:.4f}')
            print(f'Coordinate Loss: {total_coord_loss/n_samples:.4f}')
            print(f'Direction Loss: {total_dir_loss/n_samples:.4f}')
            print(f'Speed Loss: {total_speed_loss/n_samples:.4f}')
            print('-' * 90)

    return total_loss / n_samples

if __name__ == '__main__':
    # 데이터 분할
    train_data, val_data = get_data(df, 0.8) 

    # 모델 초기화
    model = TransAm(4, 3).to(device)

    # 손실 함수와 옵티마이저 설정
    criterion = nn.MSELoss()
    lr = 0.0001 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # 학습 설정
    epochs = 100
    batch_size = 32
    best_val_loss = float('inf')

    # 학습 루프
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
    
        # 학습
        train_loss = train(train_data, model, optimizer, criterion, scheduler, epoch, batch_size)
    
        # 검증
        val_loss = evaluate(model, val_data, criterion)

        scheduler.step()

        # 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'hornet_model.pth')
    
