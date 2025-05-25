import os
import argparse
import glob
import csv
from natsort import natsorted
import json

def evaluate(annotation_dir, result_dir, output_csv, fps_dict=None):
    correct_video_preds = 0
    total_video_preds = 0

    annotation_files = natsorted(glob.glob(os.path.join(annotation_dir, '*.txt')))

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Video Name', 'Has Fall (GT)', 'Fall Detected', 'Video Accuracy', 'Video Result', 'FPS']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        video_acc_sum = 0
        total_videos = 0
        fps_sum = 0.0

        for annotation_file in annotation_files:
            video_name = os.path.splitext(os.path.basename(annotation_file))[0]
            result_file = os.path.join(result_dir, f'{video_name}_result.txt')

            if not os.path.exists(result_file):
                print(f"❌ Result not found: {video_name}")
                continue

            # 讀取 annotation：兩行為 0 0 表示沒跌倒，否則為有跌倒
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2:
                    print(f"⚠️ Invalid annotation file: {annotation_file}")
                    continue
                start_frame = int(lines[0].strip())
                end_frame = int(lines[1].strip())
                has_fall = not (start_frame == 0 and end_frame == 0)

            # 讀取 result：第一行是否為 'fall'
            with open(result_file, 'r') as f:
                lines = f.readlines()
                fall_detected = lines[0].strip() == 'fall'

            is_correct = has_fall == fall_detected
            video_result = 'Correct' if is_correct else 'Incorrect'
            video_accuracy = 1 if is_correct else 0
            total_video_preds += 1
            if is_correct:
                correct_video_preds += 1

            fps = fps_dict.get(video_name, None) if fps_dict else None
            fps_str = f"{fps:.2f}" if fps is not None else '-'

            # ➕ 把準確率寫進 result 檔案最下面
            with open(result_file, 'a') as rf:
                rf.write(f"\nVideo Accuracy: {video_accuracy * 100:.2f}%\n")

            writer.writerow({
                'Video Name': video_name,
                'Has Fall (GT)': 'Yes' if has_fall else 'No',
                'Fall Detected': 'Yes' if fall_detected else 'No',
                'Video Accuracy': f"{video_accuracy:.2f}",
                'Video Result': video_result,
                'FPS': fps_str
            })

            total_videos += 1
            video_acc_sum += video_accuracy
            if fps is not None:
                fps_sum += fps

        if total_videos > 0:
            avg_fps = fps_sum / total_videos
            overall_acc = correct_video_preds / total_video_preds
            writer.writerow({
                'Video Name': 'Average FPS',
                'Has Fall (GT)': '-', 'Fall Detected': '-', 'Video Result': '-',
                'Video Accuracy': f"{overall_acc:.4f}", 'FPS': f"{avg_fps:.2f}"
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate fall detection results.')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory of result files.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the CSV report.')
    parser.add_argument('--fps_json', type=str, default=None, help='JSON file storing FPS per video')
    parser.add_argument('--dataset', type=str, choices=['le2i', 'urfd'], required=True, help='Specify dataset')
    args = parser.parse_args()

    # 根據資料集選 annotation 路徑
    if args.dataset == 'le2i':
        annotation_dir = "/home/moore/school/Le2i/all/Annotation_files"
    else:
        annotation_dir = "/home/moore/school/URFD/annotation"

    fps_dict = None
    if args.fps_json and os.path.exists(args.fps_json):
        with open(args.fps_json, 'r') as f:
            fps_dict = json.load(f)

    evaluate(annotation_dir, args.result_dir, args.output_csv, fps_dict=fps_dict)
