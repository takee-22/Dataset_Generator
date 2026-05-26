# v8: Improved filename format to prevent collisions across multiple runs.
# Filenames now include timestamp, milliseconds, and a short random hex suffix.
# Example: 20260518_143022_456_A3F2.jpg

import cv2
import os
import random
import numpy as np
from datetime import datetime
from ultralytics import YOLO

VIDEO_FOLDER      = r"C:\videos"
SAVE_DIR          = "dataset"
MODEL_PATH        = r"C:\models\best.pt"
INTERVAL          = 2
FRAME_DIFF_THRESH = 8.0
CONF_THRES        = 0.25
IMGSZ             = 960
JPEG_QUALITY      = 95

SUPPORTED = [".mp4", ".avi", ".mov", ".mkv", ".dav", ".flv", ".mpg", ".wmv"]

images_dir = os.path.join(SAVE_DIR, "images")
labels_dir = os.path.join(SAVE_DIR, "labels")
viz_dir    = os.path.join(SAVE_DIR, "viz")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(viz_dir,    exist_ok=True)


def make_filename():
    now     = datetime.now()
    dt      = now.strftime("%Y%m%d_%H%M%S")
    ms      = f"{now.microsecond // 1000:03d}"
    suffix  = ''.join(random.choices("0123456789ABCDEF", k=4))
    return f"{dt}_{ms}_{suffix}"


model  = YOLO(MODEL_PATH)
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

    frame_id  = 0
    saved     = 0
    last_gray = None

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

            result = model(frame, imgsz=IMGSZ, conf=CONF_THRES, verbose=False)[0]

            if len(result.boxes) == 0:
                frame_id += 1
                continue

            h, w = frame.shape[:2]
            label_lines = []
            viz = frame.copy()

            for box in result.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(viz, model.names[cls_id], (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            base = make_filename()

            cv2.imwrite(os.path.join(images_dir, base + ".jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            with open(os.path.join(labels_dir, base + ".txt"), "w") as f:
                f.write("\n".join(label_lines))
            cv2.imwrite(os.path.join(viz_dir, base + ".jpg"), viz,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            last_gray = gray
            saved += 1
            print(f"  Saved: {base}  |  Detections: {len(result.boxes)}  |  Total: {saved}")

        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_id += 1

    cap.release()
    print(f"  Done: {saved} frames from {video_name}\n")
    total_saved += saved

cv2.destroyAllWindows()
print(f"All videos done. Total saved: {total_saved}")
