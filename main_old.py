import os
import glob
import cv2
import time
import torch
import argparse
import numpy as np
import json
from natsort import natsorted
import sys
sys.path.append(os.path.abspath("."))

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single
from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG
from evaluation import evaluate
import torch.nn.functional as F  # 檔案開頭要有這行



def preproc(image, resize_fn):
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def kpt2bbox(kpt, ex=20):
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

def process_video(video_path, device, inp_dets, inp_pose, pose_backbone, result_dir, save_video):
    result_list = []
    fall_detected = False
    fps_list = []
    
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)
    pose_model = SPPE_FastPose(pose_backbone, inp_pose[0], inp_pose[1], device=device)
    action_model = TSSTG(
    weight_file='/home/moore/school/Human-Falling-Detect-Tracks/Actionsrecognition/saved/lite/tsstg-lite.pth' if args.lite else '/home/moore/school/Human-Falling-Detect-Tracks/Actionsrecognition/saved/ori2/tsstg-full.pth',
    use_lite=args.lite,
    device=args.device
)

    resize_fn = ResizePadding(inp_dets, inp_dets)
    if video_path == "webcam":
        cam = CamLoader_Q(0, queue_size=1000, preprocess=lambda x: preproc(x, resize_fn)).start()
        video_name = "webcam"
    else:
        cam = CamLoader_Q(video_path, queue_size=1000, preprocess=lambda x: preproc(x, resize_fn)).start()
        video_name = os.path.splitext(os.path.basename(video_path))[0]

    tracker = Tracker(max_age=30, n_init=3)

    # 設定 VideoWriter 來儲存影片
    save_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 設定編碼格式
        output_path = os.path.join(result_dir, video_name + '_output.mp4')

        fps = cam.fps
        frame = cam.getitem()  # 抓第一幀確認尺寸
        frame_size = (frame.shape[1], frame.shape[0])  # (width, height)

        save_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    f = 0
    while cam.grabbed():
        f += 1
        frame_start_time = time.time()
        
        frame = cam.getitem()
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)
        tracker.predict()

        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []
        if detected is not None:
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

        tracker.update(detections)

        falling_flag = 0
        for track in tracker.tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                out_tensor = torch.tensor(out[0])  # 轉成 tensor
                prob = F.softmax(out_tensor, dim=0)  # 機率分布
                action_idx = prob.argmax().item()
                action_name = action_model.class_names[action_idx]
                action = '{}'.format(action_name)
                pts = np.array(track.keypoints_list, dtype=np.float32)

                if action_name == 'Falling':
                    clr = (255, 0, 0)
                    falling_flag = 1
                    fall_detected = True
                elif action_name == 'Fall':
                    clr = (255, 200, 0)

            result_list.append((f, falling_flag))

            if track.time_since_update == 0:
                frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)

        frame_time = time.time() - frame_start_time
        fps_list.append(1.0 / frame_time)

        frame = cv2.putText(frame, 'Frame: {}, FPS: {:.2f}'.format(f, 1.0 / frame_time),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if save_video and save_writer is not None:
            save_writer.write(frame)

        cv2.imshow('frame', frame[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()
    
    if save_writer is not None:
        save_writer.release()

    os.makedirs(result_dir, exist_ok=True)
    result_filename = os.path.join(result_dir, video_name + '_result.txt')
    with open(result_filename, 'w') as f:
        if fall_detected:
            f.write("fall\n")
        for frame_idx, res in result_list:
            f.write(f"{frame_idx},{res}\n")
    
    avg_fps = np.mean(fps_list)
    print(f"影片 {video_path} 平均 FPS: {avg_fps:.2f}")

    return avg_fps

if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Batch Demo.')
    par.add_argument('-V', '--video', type=str, default='', help='Path to a single video file.')
    par.add_argument('-F', '--folder', type=str, default='', help='Folder path containing video files.')
    par.add_argument('-R', '--result_dir', type=str, required=True, help='Folder path to save result files.')
    par.add_argument('--detection_input_size', type=int, default=384, help='Size of input in detection model.')
    par.add_argument('--pose_input_size', type=str, default='224x160', help='Size of input in pose model.')
    par.add_argument('--pose_backbone', type=str, default='resnet50', help='Backbone model for SPPE FastPose.')
    par.add_argument('--device', type=str, default='cuda', help='Device to run model on cpu or cuda.')
    par.add_argument('--save', action='store_true', help='Save the processed video output.')
    par.add_argument('--lite', action='store_true', help='Use Lite TSSTG model')
    par.add_argument('--cam', action='store_true', help='Use webcam as input source')


    args = par.parse_args()

    if args.cam:
        video_list = ["webcam"]
    elif args.video:
        video_list = [args.video]
    elif args.folder:
        video_list = natsorted(glob.glob(os.path.join(args.folder, '*.avi')))
    else:
        raise ValueError("You must provide either --video or --folder ")

    
    print(f"🔧 使用的 TSSTG 模型：{'Lite' if args.lite else 'Full'}")
    total_fps = []
    fps_per_video = {}

    for video_path in video_list:
        print(f"開始處理影片: {video_path}")
        avg_fps = process_video(
            video_path,
            args.device,
            args.detection_input_size,
            tuple(map(int, args.pose_input_size.split('x'))),
            args.pose_backbone,
            args.result_dir,
            args.save
        )
        total_fps.append(avg_fps)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        fps_per_video[video_name] = avg_fps

    avg_fps = np.mean(total_fps)
    print(f"所有影片處理完成！平均 FPS: {avg_fps:.2f}")

    # 設定標註與報告輸出路徑
    annotation_dir = "/home/moore/school/Le2i/Lecture_room/annotation"
    output_csv = os.path.join(args.result_dir, "evaluation_report.csv")

    # 把FPS存成json
    fps_json_path = os.path.join(args.result_dir, 'fps.json')
    with open(fps_json_path, 'w') as f:
        json.dump(fps_per_video, f)

    print(f"\n🔍 開始評估結果...")
    evaluate(annotation_dir, args.result_dir, output_csv, fps_dict=fps_per_video)
    print(f"✅ 評估完成，結果已儲存到: {output_csv}")

    

    
