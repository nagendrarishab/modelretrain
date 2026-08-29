#To split video(s) into a dataset of images, choosing one of three sampling modes
# python src/videosplit.py --video-path path/to/your_video.mp4 --target-fps 30
# python src/videosplit.py --video-path video1.mp4 video2.mp4 --num-frames 200
# python src/videosplit.py --video-path video1.mp4 video2.mp4 --frames-per-minute 60

import argparse
from pathlib import Path

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--video-path", nargs="+", default=["data_vid.mp4"], help="one or more video files")
parser.add_argument("--output-dir", default=".", help="where to save extracted <video_name>_frame_NNNNN.jpg files")

mode_group = parser.add_mutually_exclusive_group()
mode_group.add_argument("--target-fps", type=int, help="extract ~N frames per second of video")
mode_group.add_argument("--num-frames", type=int, help="extract exactly N frames, evenly spaced across the whole video")
mode_group.add_argument("--frames-per-minute", type=float, help="extract N frames per minute of video")
args = parser.parse_args()


def extract_frames(video_path, output_dir):
    prefix = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    count = 0
    saved_count = 0

    if args.num_frames:
        # Evenly space the requested number of frames across the full video.
        num_frames = min(args.num_frames, total_frames) if total_frames > 0 else args.num_frames
        frame_indices = {round(i * (total_frames - 1) / max(1, num_frames - 1)) for i in range(num_frames)} if num_frames > 1 else {0}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count in frame_indices:
                cv2.imwrite(f"{output_dir}/{prefix}_frame_{saved_count:05d}.jpg", frame)
                saved_count += 1
            count += 1

        print(f"[{prefix}] Extracted {saved_count} frames evenly spaced across the video.")
    else:
        if args.frames_per_minute:
            target_fps = args.frames_per_minute / 60
        else:
            target_fps = args.target_fps or 30

        frame_interval = max(1, int(video_fps / target_fps))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                cv2.imwrite(f"{output_dir}/{prefix}_frame_{saved_count:05d}.jpg", frame)
                saved_count += 1
            count += 1

        if args.frames_per_minute:
            print(f"[{prefix}] Extracted {saved_count} frames at ~{args.frames_per_minute} frames/minute.")
        else:
            print(f"[{prefix}] Extracted {saved_count} frames at ~{target_fps} FPS.")

    cap.release()


for video_path in args.video_path:
    extract_frames(video_path, args.output_dir)
