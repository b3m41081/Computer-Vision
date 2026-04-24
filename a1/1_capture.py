import cv2
import os
from pathlib import Path

CAMERA_ID = 0
SAVE_DIR = Path("captured_images")

def ensure_save_dir():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

def next_filename():
    existing = sorted(SAVE_DIR.glob("image_*.jpg"))
    if not existing:
        return SAVE_DIR / "image_001.jpg"

    numbers = []
    for file in existing:
        try:
            numbers.append(int(file.stem.split("_")[1]))
        except:
            pass

    return SAVE_DIR / f"image_{max(numbers, default=0)+1:03d}.jpg"

def delete_all_images():
    if SAVE_DIR.exists():
        for f in SAVE_DIR.glob("*"):
            f.unlink()

def main():
    ensure_save_dir()
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = next_filename()
            cv2.imwrite(str(filename), frame)
            print("Saved:", filename)
        elif key == ord("d"):
            delete_all_images()
            print("Deleted all images")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
