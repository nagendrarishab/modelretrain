#To split the video into dataset of images based on the required frames per each second

import cv2

video_path = "data_vid.mp4"
cap = cv2.VideoCapture(video_path)

video_fps = cap.get(cv2.CAP_PROP_FPS)

target_fps = 30

frame_interval = max(1, int(video_fps / target_fps))

count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count % frame_interval == 0:
        cv2.imwrite(f"frame_{saved_count:05d}.jpg", frame)
        saved_count += 1
    count += 1

cap.release()
print(f"Extracted {saved_count} frames at ~{target_fps} FPS.")