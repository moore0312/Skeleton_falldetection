import torch
from ActionsEstLoader import TSSTG

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, total / 1e6

if __name__ == "__main__":
    model = TSSTG().model  # 初始化模型
    total, total_m = count_parameters(model)
    print(f"參數總數：{total:,} 個")
    print(f"參數總數（單位 M）：{total_m:.2f} M")
