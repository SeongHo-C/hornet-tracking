import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_synthetic_data(n_points=5000):
    start_time = datetime(2025, 1, 22, 17, 16, 28)
    times = [start_time + timedelta(seconds=i * 0.01) for i in range(n_points)]

    # 시간에 따른 단순 증가/감소 패턴
    t = np.linspace(0, 1, n_points)

    # 오른쪽으로 이동하면서 위로 올라가는 곡선
    x = t * 700 + 50  # 50에서 시작해서 750까지
    y = 100 + 400 * t - 100 * np.sin(np.pi * t)  # 곡선 운동
 
    # 화면 범위 내로 클리핑 (800x600)
    x = np.clip(x, 0, 800)
    y = np.clip(y, 0, 600)
    
    df = pd.DataFrame({
        'tracking_id': 1,
        'datetime': times,
        'x': x.round(2),
        'y': y.round(2)
    })
    
    return df

synthetic_df = generate_synthetic_data(5000)

synthetic_df.to_csv('synthetic_hornet_sequences.csv', index=False)

plt.figure(figsize=(10, 8))
plt.xlim(0, 800)
plt.ylim(0, 600)

plt.plot(synthetic_df['x'], synthetic_df['y'], color='blue', alpha=0.6, label=f'Track ID 1')

plt.title('Hornet Movement Path')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()