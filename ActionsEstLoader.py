import os
import torch
import numpy as np

from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph, StreamSpatialTemporalGraphLite
from pose_utils import normalize_points_with_size, scale_pose


class TSSTG(object):
    """Two-Stream Spatial Temporal Graph Model Loader.
    Args:
        weight_file: (str) Path to trained weights file.
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """
    def __init__(self,
                 weight_file,
                 use_lite=False,
                 device='cuda'):
        self.graph_args = {'strategy': 'spatial'}
        self.class_names = ['Standing', 'Walking', 'Sitting', 'Fall',
                            'Stand up', 'Sit down', 'Falling']
        self.num_class = len(self.class_names)
        self.device = device

        if use_lite:
            self.model = StreamSpatialTemporalGraphLite(3, self.graph_args, self.num_class).to(self.device)
        else:
            self.model = TwoStreamSpatialTemporalGraph(self.graph_args, self.num_class).to(self.device)
        self.model.load_state_dict(torch.load(weight_file))
        self.model.eval()

    def predict(self, pts, image_size):
        """
        Predict actions from single person skeleton points and score in time sequence.
        Args:
            pts: np.array of shape (T, V, C), where C = (x, y, score)
            image_size: (width, height)
        Returns:
            np.array: action classification probabilities
        """
        # 步驟 1：標準化與縮放關鍵點位置
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
        pts[:, :, :2] = scale_pose(pts[:, :, :2])

        # 步驟 2：補一個關鍵點（脖子）
        pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, 1)), axis=1)

        # 步驟 3：轉換成 PyTorch tensor，shape 變成 (N, C=3, T, V)
        pts = pts.transpose(2, 0, 1)  # (C, T, V)
        pts = torch.tensor(pts, dtype=torch.float32).unsqueeze(0)  # (1, C, T, V)

        # 步驟 4：算 mot = x, y 差值
        mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]  # → (1, 2, T-1, V)
        pts = pts[:, :, 1:, :]                        # → (1, 3, T-1, V)

        # 🩹 補 channel，只對 Lite 模型做，且只補一次（mot=2 channel 才補）
        if isinstance(self.model, StreamSpatialTemporalGraphLite) and mot.shape[1] == 2:
            zeros = torch.zeros_like(mot[:, :1, :, :])  # (1, 1, T-1, V)
            mot = torch.cat((mot, zeros), dim=1)        # → (1, 3, T-1, V)

        # 放到 GPU 或 CPU
        pts = pts.to(self.device)
        mot = mot.to(self.device)

        out = self.model((pts, mot))  # 給模型推論
        return out.detach().cpu().numpy()


