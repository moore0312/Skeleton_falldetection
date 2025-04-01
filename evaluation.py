import os
import argparse
import glob
import csv
from natsort import natsorted
import json

def read_annotation(annotation_file):
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
        start_frame = int(lines[0].strip())
        end_frame = int(lines[1].strip())
    return start_frame, end_frame

def read_result(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()
        fall_flag = lines[0].strip() == 'fall'
        result_frames = set()
        for line in lines[1:]:
            frame, flag = line.strip().split(',')
            if int(flag) == 1:
                result_frames.add(int(frame))
    return fall_flag, result_frames

def format_score(score):
    return '-' if score == '-' else f"{score:.2f}"

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

            with open(annotation_file, 'r') as f:
                start_frame = int(f.readline().strip())
                end_frame = int(f.readline().strip())
                has_fall = not (start_frame == 0 and end_frame == 0)

            with open(result_file, 'r') as f:
                lines = f.readlines()
                fall_detected = lines[0].strip() == 'fall'

            is_correct = has_fall == fall_detected
            video_result = 'Correct' if is_correct else 'Incorrect'
            video_accuracy = 1 if is_correct else 0
            total_video_preds += 1
            if is_correct:
                correct_video_preds += 1
            fps = fps_dict.get(video_name, None)
            fps_str = f"{fps:.2f}" if fps is not None else '-'

            writer.writerow({
                'Video Name': video_name,
                'Has Fall (GT)': 'Yes' if has_fall else 'No',
                'Fall Detected': 'Yes' if fall_detected else 'No',
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
                'Has Fall (GT)': '-', 'Fall Detected': '-', 'Video Result': '-','Video Accuracy': f"{overall_acc:.4f}", 'FPS': f"{avg_fps:.2f}"
            })
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate fall detection results.')
    parser.add_argument('--annotation_dir', type=str, required=True, help='Directory of annotation files.')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory of result files.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the CSV report.')
    parser.add_argument('--fps_json', type=str, default=None, help='JSON file storing FPS per video')
    args = parser.parse_args()

    # 讀json
    fps_dict = None
    if args.fps_json and os.path.exists(args.fps_json):
        with open(args.fps_json, 'r') as f:
            fps_dict = json.load(f)

    evaluate(args.annotation_dir, args.result_dir, args.output_csv, fps_dict=fps_dict)
