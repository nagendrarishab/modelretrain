#To split the video into dataset of images based on the required frames per each second
# python src/videosplit.py --video-path path/to/your_video.mp4 --target-fps 30

import argparse

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--video-path", default="data_vid.mp4")
parser.add_argument("--target-fps", type=int, default=30)
parser.add_argument("--output-dir", default=".", help="where to save extracted frame_NNNNN.jpg files")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_path)

video_fps = cap.get(cv2.CAP_PROP_FPS)

frame_interval = max(1, int(video_fps / args.target_fps))

count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count % frame_interval == 0:
        cv2.imwrite(f"{args.output_dir}/frame_{saved_count:05d}.jpg", frame)
        saved_count += 1
    count += 1

cap.release()
print(f"Extracted {saved_count} frames at ~{args.target_fps} FPS.")
