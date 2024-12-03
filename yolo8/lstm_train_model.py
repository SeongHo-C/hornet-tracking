import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class HornetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class HornetLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, output_size=2, dropout=0.3):
        super(HornetLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def prepare_data(sequence_length=30, test_size=0.2, batch_size=32):
    df = pd.read_csv('hornet_coordinates.csv')
    
    features = ['norm_x', 'norm_y', 'velocity_x', 'velocity_y', 'width', 'height']
    data = df[features].values
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i + sequence_length])
        y.append(data_scaled[i + sequence_length, :2])
    
    X = np.array(X)
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    train_dataset = HornetDataset(X_train, y_train)
    test_dataset = HornetDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, scaler

def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(test_losses, label='Validation Loss', linewidth=2)
    plt.title('Hornet Movement Prediction - Training Progress', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

def train_model(train_loader, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu', 
                epochs=1000, learning_rate=0.0005, patience=30):
    input_size = next(iter(train_loader))[0].shape[2]
    model = HornetLSTM(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=10, verbose=True)
        
    train_losses = []
    test_losses = []
    
    best_loss = float('inf')
    early_stop_counter = 0
    best_model = None
    
    print(f"Training on {device}")
    print(f"Input features: {input_size}")
    print("Model architecture:")
    print(model)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                test_loss += criterion(outputs, batch_y).item()
        
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        scheduler.step(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {train_loss:.6f}')
            print(f'Test Loss: {test_loss:.6f}')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early Stopping 체크
        if test_loss < best_loss:
            best_loss = test_loss
            early_stop_counter = 0
            best_model = model.state_dict().copy()
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss: {best_loss:.6f}")
            break
    
    # 최적의 모델 복원
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # 학습 곡선 그리기
    plot_losses(train_losses, test_losses)
    
    return model, train_losses, test_losses, best_loss

def main():
    SEQUENCE_LENGTH = 30
    BATCH_SIZE = 32
    EPOCHS = 1000
    LEARNING_RATE = 0.0005
    PATIENCE = 30
    
    print("Preparing data...")
    train_loader, test_loader, scaler = prepare_data(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
    )
    
    print("\nStarting training...")
    model, train_losses, test_losses, best_loss = train_model(
        train_loader, 
        test_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE
    )
    
    # 모델 및 관련 정보 저장
    save_dict = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'best_loss': best_loss,
        'sequence_length': SEQUENCE_LENGTH,
        'input_features': ['norm_x', 'norm_y', 'velocity_x', 'velocity_y', 'width', 'height'],
        'input_size': next(iter(train_loader))[0].shape[2],
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers
    }
    
    torch.save(save_dict, 'hornet_lstm_model.pth')
    
    print("\nTraining completed!")
    print("Model and training history saved to 'hornet_lstm_model.pth'")
    print("Loss plot saved as 'training_loss.png'")
    print(f"\nFinal best loss: {best_loss:.6f}")

if __name__ == "__main__":
    main()