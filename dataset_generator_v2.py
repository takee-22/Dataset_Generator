# Added timestamp filename and a preview window with Q to quit

import cv2
import os
import time

SOURCE   = r"C:\videos\traffic.mp4"
SAVE_DIR = "dataset"
INTERVAL = 2

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(SOURCE)
fps = cap.get(cv2.CAP_PROP_FPS)
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
        print(f"Saved: {filename}  |  Total: {saved}")

    cv2.imshow("Preview", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
print(f"\nDone. {saved} frames saved.")
