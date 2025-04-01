"""
This script to extract skeleton joints position and score.
"""

import os
import cv2
import time
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import sys
sys.path.append(os.path.abspath(".."))

from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from fn import vis_frame_fast

save_path = '/home/moore/school/Human-Falling-Detect-Tracks/Data/Le2i-pose+score.csv'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

annot_file = '/home/moore/school/Human-Falling-Detect-Tracks/Data/coffee_all.csv'
video_folder = '/home/moore/school/Le2i/all/video'
annot_folder = '/home/moore/school/Le2i/all/Annotation_files'

# DETECTION MODEL
detector = TinyYOLOv3_onecls()

# POSE MODEL
inp_h = 320
inp_w = 256
pose_estimator = SPPE_FastPose(backbone='resnet50', input_height=inp_h, input_width=inp_w)

columns = ['video', 'frame', 'Nose_x', 'Nose_y', 'Nose_s', 'LShoulder_x', 'LShoulder_y', 'LShoulder_s',
           'RShoulder_x', 'RShoulder_y', 'RShoulder_s', 'LElbow_x', 'LElbow_y', 'LElbow_s', 'RElbow_x',
           'RElbow_y', 'RElbow_s', 'LWrist_x', 'LWrist_y', 'LWrist_s', 'RWrist_x', 'RWrist_y', 'RWrist_s',
           'LHip_x', 'LHip_y', 'LHip_s', 'RHip_x', 'RHip_y', 'RHip_s', 'LKnee_x', 'LKnee_y', 'LKnee_s',
           'RKnee_x', 'RKnee_y', 'RKnee_s', 'LAnkle_x', 'LAnkle_y', 'LAnkle_s', 'RAnkle_x', 'RAnkle_y',
           'RAnkle_s', 'label']

def normalize_points_with_size(points_xy, width, height, flip=False):
    points_xy[:, 0] /= width
    points_xy[:, 1] /= height
    if flip:
        points_xy[:, 0] = 1 - points_xy[:, 0]
    return points_xy

pose_annot = pd.read_csv(annot_file)
pose_annot['video'] = pose_annot['video'].apply(lambda x: x if x.endswith('.avi') else x + '.avi')
vid_list = pose_annot['video'].unique()

for vid in vid_list:
    

    print(f'Process on: {vid}')
    df = pd.DataFrame(columns=columns)
    cur_row = 0

    # pose labels
    frames_label = pose_annot[pose_annot['video'] == vid].reset_index(drop=True)
    print(f"[DEBUG] 處理中影片：{vid}，找到標註筆數：{len(frames_label)}")
    cap = cv2.VideoCapture(os.path.join(video_folder, vid))
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # bbox 標註
    annot_path = os.path.join(annot_folder, os.path.splitext(vid)[0] + ".txt")
    bbox_annot = None
    if os.path.exists(annot_path):
        bbox_annot = pd.read_csv(annot_path, header=None,
                                 names=['frame_idx', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
        bbox_annot = bbox_annot.dropna().reset_index(drop=True)
        assert frames_count == len(bbox_annot), 'frame count not equal! {} and {}'.format(frames_count, len(bbox_annot))

    

    fps_time = 0
    i = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label_row = frames_label.loc[frames_label['frame'] == i, 'label']
        cls_idx = int(label_row.values[0]) if not label_row.empty else 0

        if bbox_annot is not None and not bbox_annot.empty:
            bb = np.array(bbox_annot.iloc[i-1, 2:].astype(int))
        else:
            bb = detector.detect(frame)[0, :4].numpy().astype(int)
        bb[:2] = np.maximum(0, bb[:2] - 5)
        bb[2:] = np.minimum(frame_size, bb[2:] + 5) if bb[2:].any() != 0 else bb[2:]

        result = []
        if bb.any() != 0:
            result = pose_estimator.predict(frame, torch.tensor(bb[None, ...]),
                                            torch.tensor([[1.0]]))

        if len(result) > 0:
            pt_norm = normalize_points_with_size(result[0]['keypoints'].numpy().copy(),
                                                 frame_size[0], frame_size[1])
            pt_norm = np.concatenate((pt_norm, result[0]['kp_score']), axis=1)
            row = [vid, i, *pt_norm.flatten().tolist(), cls_idx]
            scr = result[0]['kp_score'].mean()
        else:
            row = [vid, i, *[np.nan] * (13 * 3), cls_idx]
            scr = 0.0

        df.loc[cur_row] = row
        cur_row += 1

        # 可視化（可選）
        frame = vis_frame_fast(frame, result)
        frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
        frame = cv2.putText(frame, f'Frame: {i}, Pose: {cls_idx}, Score: {scr:.4f}',
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        frame = frame[:, :, ::-1]
        fps_time = time.time()
        i += 1

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if os.path.exists(save_path):
        df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        df.to_csv(save_path, mode='w', index=False)
    
    print(f"✅ 輸出完成，共寫入 {len(df)} 筆資料至：{save_path}")
