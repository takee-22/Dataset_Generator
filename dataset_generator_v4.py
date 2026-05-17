# v4: Added support for processing multiple videos from a folder.
# Script loops through all video files in VIDEO_FOLDER automatically.
# Supported formats: mp4, avi, mov, mkv, dav, flv, mpg, wmv.

import cv2
import os
import time

VIDEO_FOLDER = r"C:\videos"
SAVE_DIR     = "dataset"
INTERVAL     = 2

SUPPORTED = [".mp4", ".avi", ".mov", ".mkv", ".dav", ".flv", ".mpg", ".wmv"]

os.makedirs(SAVE_DIR, exist_ok=True)

videos = [f for f in os.listdir(VIDEO_FOLDER) if os.path.splitext(f)[1].lower() in SUPPORTED]

if not videos:
    print("No video files found.")
    exit()

print(f"Found {len(videos)} video(s).\n")

total_saved = 0

for video_name in videos:
    video_path = os.path.join(VIDEO_FOLDER, video_name)
    print(f"Processing: {video_name}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    stride = int(fps * INTERVAL)

    frame_id = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % stride == 0:
            filename = time.strftime("frame_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(os.path.join(SAVE_DIR, filename), frame)
            saved += 1
            print(f"  Saved: {filename}  |  Total: {saved}")

        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_id += 1

    cap.release()
    print(f"  Done: {saved} frames from {video_name}\n")
    total_saved += saved

cv2.destroyAllWindows()
print(f"All videos done. Total saved: {total_saved}")
