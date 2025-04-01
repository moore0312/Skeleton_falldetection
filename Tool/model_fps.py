import torch
import time
import numpy as np
from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph
from Actionsrecognition.Utils import Graph

T, V = 30, 14
pts = torch.rand(1, 3, T, V).cuda()

# 不要補第0幀，不要補score
mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]  # → (1, 2, 29, 14)


# 初始化模型
graph_args = {'strategy': 'spatial'}
model = TwoStreamSpatialTemporalGraph(graph_args, num_class=7).cuda()
model.eval()

# 預熱 CUDA
for _ in range(10):
    _ = model((pts, mot))

# 正式測量 100 次推論時間
times = []
with torch.no_grad():
    for _ in range(100):
        start = time.time()
        _ = model((pts, mot))
        end = time.time()
        times.append(end - start)

avg_time = np.mean(times)
fps = 1.0 / avg_time

print(f"✅ 平均單筆推論時間：{avg_time:.6f} 秒")
print(f"🚀 模型理論 FPS：{fps:.2f} frame/s")
