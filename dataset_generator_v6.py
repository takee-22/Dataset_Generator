# v6: Added duplicate frame filter using frame difference.
# Frames that look too similar to the last saved frame are skipped.
# FRAME_DIFF_THRESH controls sensitivity — higher = more duplicates allowed.

import cv2
import os
import time
import numpy as np

VIDEO_FOLDER    = r"C:\videos"
SAVE_DIR        = "dataset"
INTERVAL        = 2
FRAME_DIFF_THRESH = 8.0

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
    video_stem = os.path.splitext(video_name)[0]
    out_dir    = os.path.join(SAVE_DIR, video_stem)

    os.makedirs(out_dir, exist_ok=True)
    print(f"Processing: {video_name}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    stride = int(fps * INTERVAL)

    frame_id   = 0
    saved      = 0
    last_gray  = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % stride == 0:
            small = cv2.resize(frame, (320, 180))
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if last_gray is not None:
                diff = float(np.mean(cv2.absdiff(gray, last_gray)))
                if diff < FRAME_DIFF_THRESH:
                    frame_id += 1
                    continue

            filename = time.strftime("frame_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(os.path.join(out_dir, filename), frame)
            last_gray = gray
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
