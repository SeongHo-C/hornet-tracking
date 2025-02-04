import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(n_points=5000):
    start_time = datetime(2025, 1, 22, 17, 16, 28)

    times = [start_time + timedelta(seconds=i * 0.01) for i in range(n_points)]

    # 시간 기반 각도 생성 (라디안)
    t = np.linspace(0, 20*np.pi, n_points)  # 10번의 완전한 사이클
    
    # x 좌표: 화면 중앙(400)을 기준으로 sin 파형
    x_amplitude = 300  # 진폭
    x_offset = 400    # 화면 중앙
    x = x_offset + x_amplitude * np.sin(t)

    # y 좌표: 화면 중앙(300)을 기준으로 두 배 빠른 주기의 sin 파형
    y_amplitude = 200
    y_offset = 300
    y = y_offset + y_amplitude * np.sin(2*t)
    
    # 화면 범위 내로 클리핑 (800x600)
    x = np.clip(x, 0, 800)
    y = np.clip(y, 0, 600)
    
    # DataFrame 생성
    df = pd.DataFrame({
        'tracking_id': 1,
        'datetime': times,
        'x': x,
        'y': y
    })
    
    # 소수점 2자리까지 반올림
    df['x'] = df['x'].round(2)
    df['y'] = df['y'].round(2)
    
    return df

# 데이터 생성
synthetic_df = generate_synthetic_data(5000)

# CSV 파일로 저장
synthetic_df.to_csv('synthetic_hornet_sequences2.csv', index=False)

# 생성된 데이터 확인
print("Generated data sample:")
print(synthetic_df.head())
print("\nShape:", synthetic_df.shape)
print("\nValue ranges:")
print(f"X: {synthetic_df['x'].min():.2f} to {synthetic_df['x'].max():.2f}")
print(f"Y: {synthetic_df['y'].min():.2f} to {synthetic_df['y'].max():.2f}")