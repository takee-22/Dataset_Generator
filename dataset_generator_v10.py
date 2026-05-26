# v10: ROI can now be drawn interactively on the first frame using mouse clicks.
# Left click to add points, right click to undo, C to confirm, R to reset.
# Confirmed ROI is saved to roi.txt and reloaded automatically on next run.

import cv2
import os
import random
import numpy as np
from datetime import datetime
from ultralytics import YOLO

VIDEO_FOLDER      = r"C:\videos"
SAVE_DIR          = "dataset"
MODEL_PATH        = r"C:\models\best.pt"
ROI_FILE          = "roi.txt"
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
    now    = datetime.now()
    dt     = now.strftime("%Y%m%d_%H%M%S")
    ms     = f"{now.microsecond // 1000:03d}"
    suffix = ''.join(random.choices("0123456789ABCDEF", k=4))
    return f"{dt}_{ms}_{suffix}"


def save_roi(points):
    with open(ROI_FILE, "w") as f:
        for x, y in points:
            f.write(f"{x},{y}\n")


def load_roi():
    if not os.path.exists(ROI_FILE):
        return []
    points = []
    with open(ROI_FILE) as f:
        for line in f:
            x, y = line.strip().split(",")
            points.append((int(x), int(y)))
    return points


def draw_roi(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if param:
            param.pop()


def get_roi_from_user(frame):
    points = []
    clone  = frame.copy()

    cv2.namedWindow("Draw ROI")
    cv2.setMouseCallback("Draw ROI", draw_roi, points)
    print("Draw ROI: Left click to add points | Right click to undo | C to confirm | R to reset")

    while True:
        display = clone.copy()
        for pt in points:
            cv2.circle(display, pt, 5, (0, 255, 255), -1)
        if len(points) >= 2:
            cv2.polylines(display, [np.array(points, dtype=np.int32)],
                          isClosed=False, color=(0, 255, 255), thickness=2)
        if len(points) >= 3:
            cv2.polylines(display, [np.array(points, dtype=np.int32)],
                          isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.putText(display, "C: Confirm  |  R: Reset  |  Right click: Undo",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Draw ROI", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c") and len(points) >= 3:
            break
        elif key == ord("r"):
            points.clear()

    cv2.destroyWindow("Draw ROI")
    return points


def in_roi(cx, cy, contour):
    if contour is None:
        return True
    return cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0


model  = YOLO(MODEL_PATH)
videos = [f for f in os.listdir(VIDEO_FOLDER) if os.path.splitext(f)[1].lower() in SUPPORTED]

if not videos:
    print("No video files found.")
    exit()

print(f"Found {len(videos)} video(s).\n")

# Load or draw ROI once using the first frame of the first video
roi_points = load_roi()

if roi_points:
    print(f"Loaded ROI from {ROI_FILE} ({len(roi_points)} points)")
else:
    cap_tmp = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, videos[0]))
    ret, sample_frame = cap_tmp.read()
    cap_tmp.release()

    if ret:
        roi_points = get_roi_from_user(sample_frame)
        save_roi(roi_points)
        print(f"ROI saved to {ROI_FILE}")
    else:
        print("Could not read first frame for ROI. Running without ROI.")

roi_contour = np.array(roi_points, dtype=np.int32) if len(roi_points) >= 3 else None

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

            h, w   = frame.shape[:2]
            label_lines = []
            viz    = frame.copy()

            if roi_contour is not None:
                cv2.polylines(viz, [roi_contour], isClosed=True, color=(0, 255, 255), thickness=2)

            for box in result.boxes:
                cls_id      = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy      = (x1 + x2) / 2, (y1 + y2) / 2

                if not in_roi(cx, cy, roi_contour):
                    continue

                bx = cx / w
                by = cy / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                label_lines.append(f"{cls_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(viz, model.names[cls_id], (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            if not label_lines:
                frame_id += 1
                continue

            base = make_filename()

            cv2.imwrite(os.path.join(images_dir, base + ".jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            with open(os.path.join(labels_dir, base + ".txt"), "w") as f:
                f.write("\n".join(label_lines))
            cv2.imwrite(os.path.join(viz_dir, base + ".jpg"), viz,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            last_gray = gray
            saved += 1
            print(f"  Saved: {base}  |  Detections: {len(label_lines)}  |  Total: {saved}")

        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_id += 1

    cap.release()
    print(f"  Done: {saved} frames from {video_name}\n")
    total_saved += saved

cv2.destroyAllWindows()
print(f"All videos done. Total saved: {total_saved}")
