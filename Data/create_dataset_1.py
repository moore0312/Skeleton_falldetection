
import os
import cv2
import pandas as pd
import re

class_names = ['Standing', 'Walking', 'Sitting', 'Fall', 'Stand up', 'Sit down', 'Falling']

def load_fall_annotation(txt_path, total_frames):
    label_map = {}
    if not os.path.exists(txt_path):
        return label_map
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        try:
            start = int(lines[0].strip())
            end = int(lines[1].strip())
            if start == 0 and end == 0:
                return {}
            if start > 0:
                for fidx in range(start, end + 1):
                    label_map[fidx] = 7  # Falling
            for fidx in range(max(1, end + 1), total_frames + 1):
                label_map[fidx] = 4  # Fall
        except:
            return {}
    return label_map

video_folder = '/home/moore/school/Le2i/Home_02/Home_02/Videos'

def natural_key(filename):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', filename)]

video_list = sorted(
    [f for f in os.listdir(video_folder) if f.endswith('.avi')],
    key=natural_key
)

output_csv = '../Data/home02.csv'

if os.path.exists(output_csv):
    existing_df = pd.read_csv(output_csv)
else:
    existing_df = pd.DataFrame()

final_annot = []

video_index = 0
while video_index < len(video_list):
    vid = video_list[video_index]
    print(f"🎥 現在開始標註影片：{vid}")
    video_file = os.path.join(video_folder, vid)
    cap = cv2.VideoCapture(video_file)
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not existing_df.empty:
        prev_annot = existing_df[existing_df['video'] == vid].copy()
        if not prev_annot.empty:
            annot = prev_annot.sort_values('frame').reset_index(drop=True)
        else:
            annot = pd.DataFrame({'video': [vid]*frames_count,
                                  'frame': list(range(1, frames_count+1)),
                                  'label': [0]*frames_count})
    else:
        annot = pd.DataFrame({'video': [vid]*frames_count,
                              'frame': list(range(1, frames_count+1)),
                              'label': [0]*frames_count})
    cap.release()

    annot_txt = os.path.join("/home/moore/school/Le2i/Home_02/Home_02/Annotation_files", os.path.splitext(vid)[0] + ".txt")
    fall_ranges = load_fall_annotation(annot_txt, frames_count)
    if fall_ranges:
        falling_frames = [f for f, label in fall_ranges.items() if label == 7]
        fall_frames = [f for f, label in fall_ranges.items() if label == 4]
        if falling_frames:
            print(f"⚠️ Falling 範圍：{falling_frames[0]} → {falling_frames[-1]}")
        if fall_frames:
            print(f"⚠️ Fall 範圍：{fall_frames[0]} → {fall_frames[-1]}")

        for fidx, label in fall_ranges.items():
            if 0 < fidx <= len(annot):
                old_label = int(annot.at[fidx - 1, 'label'])
                if old_label == 0:
                    annot.at[fidx - 1, 'label'] = label

    cap = cv2.VideoCapture(video_file)
    active_label = None  # 新增：記住當前要連續標註的類別
    i = 0
    while cap.isOpened() and i < frames_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        label_new = int(annot.iloc[i, -1])
        cls_name_new = class_names[label_new - 1] if label_new > 0 else "-"

        if not existing_df.empty:
            old_row = existing_df[(existing_df['video'] == vid) & (existing_df['frame'] == i+1)]
            if not old_row.empty:
                label_old = int(old_row['label'].values[0])
                cls_name_old = class_names[label_old - 1] if label_old > 0 else "-"
            else:
                cls_name_old = "-"
        else:
            cls_name_old = "-"

        frame = cv2.putText(frame, f"Frame {i+1}/{frames_count}", (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        frame = cv2.putText(frame, f"new: {cls_name_new}", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        frame = cv2.putText(frame, f"old: {cls_name_old}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('a') and i > 0:
            i -= 1
        elif key == ord('d') and i < frames_count - 1:
            i += 1
            if active_label is not None:
                annot.at[i, 'label'] = active_label  # 自動標註連續幀
        elif key == ord('z'):
            active_label = None
            print("🧹 已取消連續標註")
        elif key == ord('s'):
            print(f"✅ 儲存目前影片標註 {vid}")
            if os.path.exists(output_csv):
                existing_df = pd.read_csv(output_csv)
                existing_df = existing_df[existing_df['video'] != vid]
                updated_df = pd.concat([existing_df, annot], ignore_index=True)
                updated_df.to_csv(output_csv, index=False)
            else:
                annot.to_csv(output_csv, index=False)
        elif key == 13:  # Enter 鍵
            print(f"✅ 完成 {vid}，進入下一部影片")
            final_annot.append(annot)
            break
        elif key in [ord(str(n)) for n in range(0, 8)]:
            label_value = int(chr(key))
            annot.at[i, 'label'] = label_value
            active_label = label_value  # 啟用連續標註

    cap.release()
    video_index += 1

cv2.destroyAllWindows()

if os.path.exists(output_csv):
    existing_df = pd.read_csv(output_csv)
    all_videos = pd.concat(final_annot, ignore_index=True)['video'].unique()
    existing_df = existing_df[~existing_df['video'].isin(all_videos)]
    final_annot.append(existing_df)

df_sorted = pd.concat(final_annot, ignore_index=True)

df_sorted = df_sorted.sort_values(
    by=['video', 'frame'],
    key=lambda col: col.astype(str).str.extract(r'(\d+)').astype(int)[0] if col.name == 'video' else col
)

df_sorted.to_csv(output_csv, index=False)

print(f"✅ 全部影片標註完成，已儲存至 {output_csv}")
