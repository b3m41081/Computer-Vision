import cv2
import numpy as np
from pathlib import Path

IMAGE_DIR = Path("captured_images")
PATTERN_SIZE = (9, 6)

def main():
    images = sorted(IMAGE_DIR.glob("*.jpg"))

    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    image_size = None

    for fname in images:
        img = cv2.imread(str(fname))

        if img is None:
            print(f"Bild konnte nicht geladen werden: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            cv2.drawChessboardCorners(img, PATTERN_SIZE, corners, ret)
            cv2.imshow("img", img)
            cv2.waitKey(200)
        else:
            print(f"Schachbrett nicht erkannt in: {fname}")

    cv2.destroyAllWindows()

    if not objpoints or image_size is None:
        print("Keine Schachbrett-Ecken gefunden. Kalibrierung nicht möglich.")
        return

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    np.savez(
        "calibration.npz",
        camera_matrix=mtx,
        dist_coeffs=dist,
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object),
        reprojection_error=ret
    )

    

    print("Kameramatrix (camera matrix):")
    print(mtx)

    print("\nVerzerrungskoeffizienten (distortion coefficients):")
    print(dist)

    

if __name__ == "__main__":
    main()