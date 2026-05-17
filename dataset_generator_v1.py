import cv2
import os

SOURCE   = r"C:\videos\traffic.mp4"
SAVE_DIR = "dataset"
INTERVAL = 2  

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(SOURCE)
fps = cap.get(cv2.CAP_PROP_FPS)
stride = int(fps * INTERVAL)

frame_id = 0
saved    = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % stride == 0:
        filename = f"frame_{frame_id:07d}.jpg"
        cv2.imwrite(os.path.join(SAVE_DIR, filename), frame)
        saved += 1
        print(f"Saved {filename}")

    frame_id += 1

cap.release()
print(f"\nDone. {saved} frames saved.")