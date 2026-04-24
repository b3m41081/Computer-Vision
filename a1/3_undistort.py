import cv2
import numpy as np

data = np.load("calibration.npz")
mtx = data["camera_matrix"]
dist = data["dist_coeffs"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    dst = cv2.undistort(frame, mtx, dist)

    combined = np.hstack((frame, dst))
    cv2.imshow("Original | Undistorted", combined)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
