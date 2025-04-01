import pickle
import numpy as np

# 替換成你的標籤檔案路徑
pkl_path = '/home/moore/school/Human-Falling-Detect-Tracks/Data/Le2i.pkl'

# 類別名稱（你原本的七類）
class_names = ['Standing', 'Walking', 'Sitting', 'Fall', 'Stand up', 'Sit down', 'Falling']

# 讀取資料
with open(pkl_path, 'rb') as f:
    features, labels = pickle.load(f)

# 統計每個類別的數量（labels 是 one-hot 編碼）
label_counts = np.sum(labels, axis=0)

# 顯示每類別對應的樣本數
for idx, (name, count) in enumerate(zip(class_names, label_counts)):
    print(f"{idx:2d}. {name:<10}：{int(count)} samples")
