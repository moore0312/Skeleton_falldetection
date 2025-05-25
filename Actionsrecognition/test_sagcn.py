import torch
from ST_SAGCN import ST_SAGCN

# 模型參數（依據 Le2i 設定）
num_joints = 14
seq_len = 30
num_classes = 7
in_channels = 3

# 建立模型
model = ST_SAGCN(in_channels=in_channels,
                 num_joints=num_joints,
                 num_classes=num_classes,
                 seq_len=seq_len,
                 embed_dim=64,
                 num_heads=4)

# 模擬輸入資料：N=2, C=3, T=30, V=14
x_joint = torch.randn(2, 3, seq_len, num_joints)  # (N, C, T, V)
x_vel = torch.randn(2, 3, seq_len, num_joints)

# Forward 測試
output = model(x_joint, x_vel)

print("輸出 shape:", output.shape)
print("預測 logits:", output)
