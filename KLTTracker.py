# KLTTracker.py
import cv2
import numpy as np

class KLTTracker:
    def __init__(self):
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2, #金字塔層數
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) #終止條件
        self.prev_gray = None #上一幀的灰階影像。
        self.prev_points = None #上一幀追蹤的關鍵點座標

    def initialize(self, frame, keypoints):
        """初始化 KLT 追蹤器"""
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_points = np.array(keypoints, dtype=np.float32).reshape(-1, 1, 2) #要追蹤的初始關鍵點

    def track(self, frame):
        """追蹤骨架關鍵點並返回速度和方向"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points, None, **self.lk_params)

        # 計算移動向量
        movement = next_points - self.prev_points

        # 計算垂直方向的速度（y軸）
        vertical_movement = movement[:, 0, 1]

        # 更新狀態
        self.prev_gray = gray.copy()
        self.prev_points = next_points.reshape(-1, 1, 2)

        return next_points, vertical_movement
