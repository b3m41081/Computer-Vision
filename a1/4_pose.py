import cv2
import numpy as np

PATTERN_SIZE = (9, 6)

data = np.load("calibration.npz")
mtx = data["camera_matrix"]
dist = data["dist_coeffs"]

objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)

axis = np.float32([
    [3, 0, 0],
    [0, 3, 0],
    [0, 0, -3]
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)

    if found:
        
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        
        cv2.drawChessboardCorners(frame, PATTERN_SIZE, corners2, found)

      
        success, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)

        if success:
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)

            corner = tuple(corners2[0].ravel().astype(int))
            imgpts = imgpts.astype(int)

            frame = cv2.line(frame, corner, tuple(imgpts[0].ravel()), (0, 0, 255), 3)   
            frame = cv2.line(frame, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 3)   
            frame = cv2.line(frame, corner, tuple(imgpts[2].ravel()), (255, 0, 0), 3)   

    cv2.imshow("Pose + Chessboard Corners", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()