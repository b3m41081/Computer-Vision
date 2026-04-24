# 3DCV Workshop 01 - Webcam Calibration with OpenCV

## Workshop Overview

In this workshop, you will learn how to calibrate a webcam using OpenCV. This involves four key steps:

1. **Capturing images** of a chessboard pattern.
2. **Computing and storing calibration parameters** using those images.
3. **Applying the calibration** to correct distortion in the webcam feed.
4. **Rendering a 3D coordinate system into the image.**

### Hints

- This tutorial is based on the OpenCV tutorial on [Camera Calibration](https://docs.opencv.org/4.11.0/dc/dbb/tutorial_py_calibration.html). If you are stuck with a task, you can check the solution from the tutorial.
- It make sense to write one Python file for each step.
- Fork this repository and commit your code regularly. You can also share your code with other students to get feedback and help.

### Preparation

- We are using Python 3.12 or later. Install from [python.org](https://www.python.org/downloads/)
- It is recommended to use [Visual Studio Code](https://code.visualstudio.com/download) for development, but you can also use any other suitable IDE.
- Take care that Python is properly installed by checking the version in your command line tool: ```python --version```
- Create a virtual environment so that the needed modules are only used for this project: ```python -m venv .venv```
- Activate the virtual environment **before** installing the modules:

```shell
# For Windows use
.venv\Scripts\Activate.ps1

# For Linux and Mac OS
source .venv/bin/activate
```

- **Note for Windows:** If activation fails, please consult the link in the error message (something about PowerShell ExecutionPolicy) or the following [guide](https://docs.python.org/3/library/venv.html) for more information. When the environment is activated, you should see this as a prefix on your terminal.
- Install OpenCV Python with ```pip install opencv-python```. Note that this tutorial was tested with version `opencv-python-4.11.0.86`
- Connect the webcam to your computer.
- Check if everything is set up correctly by running this Python script:

```python
import cv2

# Connect to the webcam
id = 0
cap = cv2.VideoCapture(id)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a window to display the webcam feed
cv2.namedWindow("Webcam")

# Main loop to capture images
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Show the webcam feed
    cv2.imshow("Webcam", frame)

    # Handle key presses
    key = cv2.waitKey(10)
    match key:
        case 113: # ord('q') = 113 (quit)
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
```

#### Error handling

- `Error: Could not open webcam.` might appear if the `id` is wrong or the camera is not properly connected.
- `Error: Failed to capture image.` might appear if the camera is broken. Try another one.

---

## Step 1: Capturing Images for Calibration

Before calibrating a camera, we need multiple images of a **chessboard pattern** taken from different angles. These images help OpenCV determine how the camera distorts the view.

### Task

Extend the given Python script so that it:

- Captures an image when the user presses the `'s'` key.
- Saves the images in a folder (`captured_images`).
- Allows deleting all existing images with the `'d'` key.

### Hints

- Use [`cv2.imwrite()`](https://docs.opencv.org/4.11.0/d4/da8/group__imgcodecs.html#ga8ac397bd09e48851665edbe12aa28f25) to save images.
- Use [`cv2.waitKey()`](https://docs.opencv.org/4.11.0/d7/dfc/group__highgui.html#ga5628525ad33f52eab17feebcfba38bd7) to detect key presses.
- If you are not able to capture any images, you can use the example images from OpenCV: [left01.jpg ... left14.jpg](https://github.com/opencv/opencv/blob/4.x/samples/data/left01.jpg).

- [ ] **Challenge:** Modify the script to show all saved images in a grid format.

---

## Step 2: Calibrating the Camera

Now that we have images of a chessboard, we need to compute the camera's **intrinsic parameters** (like focal length and distortion coefficients).

### How It Works

1. **Detect chessboard corners** in the images using [`cv2.findChessboardCorners()`](https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a).
2. **Refine corner detection** for accuracy using [`cv2.cornerSubPix()`](https://docs.opencv.org/4.11.0/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e).
3. **Compute camera calibration** using [`cv2.calibrateCamera()`](https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d).
4. **Save calibration parameters** to a file using [`np.savez()`](https://numpy.org/doc/2.2/reference/generated/numpy.savez.html).

### Task

Write a script that:

- Loads all the captured chessboard images.
- Detects chessboard corners and displays them for human inspection.
- Computes the camera matrix and distortion coefficients.
- Saves the results to `calibration.npz`.

### Hints

- Chessboard pattern size: `6x9` (rows x columns).
- Use [`np.mgrid`](https://numpy.org/doc/2.2/reference/generated/numpy.mgrid.html) to generate a 3D representation of chessboard points. Note that you can measure the exact corner positions in metric values or simply use the size of one chessboard field as 1.
- Use [`cv2.drawChessboardCorners()`](https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga6a10b0bb120c4907e5eabbcd22319022) to visualize the corners. Human inspection means simply that you check if all corners have been detected properly.

- [ ] **Challenge:** Compute and display the re-projection error after calibration. What does the error mean? Compare the values other students get or how it changes using less or more images or even bad images with blurred corners.

---

## Step 3: Undistorting the Webcam Feed

Now that we have calibration data, we can **undistort** the live webcam feed.

### How It Works

1. Load the calibration parameters (`calibration.npz`).
2. Capture a frame from the webcam.
3. Use [`cv2.undistort()`](https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga69f2545a8b62a6b0fc2ee060dc30559d) to remove distortion.
4. Display both the original and undistorted images.

### Task

Write a script that:

- Loads the saved calibration data.
- Opens the webcam and captures frames.
- Applies [`cv2.undistort()`](https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga69f2545a8b62a6b0fc2ee060dc30559d) to remove distortion.
- Displays both the **original** and **undistorted** frames side by side.

### Hints

- Use [`np.load()`](https://numpy.org/doc/2.2/reference/generated/numpy.load.html) to load saved parameters.
- Use [`cv2.imshow()`](https://docs.opencv.org/4.11.0/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563) to display multiple frames at once.

- [ ] **Challenge:** Display a **difference image** that highlights distortion effects before and after correction.

---

## Step 4: Rendering a 3D Coordinate System into the Image

With the camera calibrated, we can render a **3D coordinate system** into the image by detecting the chessboard and overlaying 3D axes.

### How It Works

1. Load the calibration parameters (`calibration.npz`).
2. Detect chessboard corners in the live feed.
3. Compute the camera's position using [`cv2.solvePnP()`](https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d).
4. Project 3D coordinate axes onto the image using [`cv2.projectPoints()`](https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c).
5. Draw the 3D axes onto the image.

### Task

Write a script that:

- Loads the calibration data.
- Detects a chessboard in the webcam feed.
- Computes the camera pose using [`cv2.solvePnP()`](https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d).
- Projects and draws a 3D coordinate system onto the image.

### Hints

- The axes should be of size `3x3` units in world coordinates.
- Use [`cv2.line()`](https://docs.opencv.org/4.11.0/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2) to draw the X, Y, and Z axes in red, green, and blue.
- This task is based on the next OpenCV tutorial: [Pose Estimation](https://docs.opencv.org/4.11.0/d7/d53/tutorial_py_pose.html).

- [ ] **Challenge:** Modify the script to overlay a **virtual 3D object** (e.g., a cube) on top of the chessboard.

---

## **Final Thoughts**

By completing this workshop, you have:

- [ ] Captured images for calibration.
- [ ] Computed and stored camera calibration parameters.
- [ ] Used those parameters to correct distortion in real time.
- [ ] Rendered a 3D coordinate system into the image.

### **Extra Challenge**

- [ ] Try calibrating with a different calibration pattern (e.g., [circles](https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/doc/acircles_pattern.png)) and observe how it affects the results.
