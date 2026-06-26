# Computer Vision

This repository contains exercises and implementations from the master's course
in Computer Vision. The focus is on classical 3D computer vision with OpenCV:
camera calibration, undistortion, pose estimation, single-view height
measurement, and stereo reconstruction.

## Contents

```text
.
├── a1/   Camera Calibration
├── a2/   Single-View Height Measurement
├── a3/   Simple Stereo
├── a4/   Own Stereo Rectification
└── a5/   3D Reconstruction with COLMAP and DUSt3R
```

### A1: Camera Calibration

Workshop 1 covers webcam calibration using a chessboard pattern.

- capture images from a webcam
- estimate the camera matrix and distortion coefficients
- undistort the live camera image
- estimate camera pose and project 3D axes into the image

Run from the `a1/` directory:

```bash
python3 1_capture.py
python3 2_calibrate.py
python3 3_undistort.py
python3 4_pose.py
```

More details are available in [`a1/README.md`](a1/README.md).

### A2: Single-View Height Measurement

Workshop 2 implements interactive height measurement from a single image. By
clicking reference points, estimating vanishing points, and entering a known
reference height, the application computes the height of a target object.

Run from the project root:

```bash
.venv/bin/python a2/src/main.py
```

More usage details are available in [`a2/README.md`](a2/README.md).

### A3: Simple Stereo

Workshop 3 computes a disparity map from a stereo image pair, evaluates it
against the ground truth, derives a depth map, and exports a colored point
cloud in PLY format. In addition to the core script, the project includes a
small browser UI for testing and saving parameter configurations.

Run from the project root:

```bash
.venv/bin/python a3/src/simple_stereo.py
```

Browser UI:

```bash
.venv/bin/python a3/src/web_ui.py
```

More details are available in [`a3/README.md`](a3/README.md).

### A4: Own Stereo Rectification

Workshop 4 estimates epipolar geometry from feature matches, rectifies an
uncalibrated stereo pair, computes disparity on the rectified images, and
derives a relative or metric depth map.

Run from the project root:

```bash
.venv/bin/python a4/src/own_stereo_rectification.py
```

More details are available in [`a4/README.md`](a4/README.md).

### A5: 3D Reconstruction

Workshop 5 reconstructs a captured scene from prepared image files. The active
pipeline first runs COLMAP for sparse structure-from-motion and then runs
DUSt3R locally on macOS for the learned 3D reconstruction result.

Run from the project root:

```bash
./a5/start_native_macos.sh
./a5/start_dust3r_macos.sh --dust3r-repo ../dust3r
```

More details are available in [`a5/README.md`](a5/README.md).

## Requirements

- Python 3
- OpenCV
- NumPy
- COLMAP for `a5`
- a webcam for the live examples in `a1`

Install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After that, A2 can be started directly in the same terminal:

```bash
.venv/bin/python a2/src/main.py
```

## Project Structure

```text
a1/
├── 1_capture.py          # capture calibration images
├── 2_calibrate.py        # compute camera calibration
├── 3_undistort.py        # live image undistortion
├── 4_pose.py             # pose estimation with projected 3D axes
├── calibration.npz       # saved calibration data
└── captured_images/      # captured chessboard images

a2/
├── img/                  # input and output images
└── src/                  # interactive height measurement

a3/
├── data/                 # stereo calibration data
├── images/               # stereo images and ground truth
├── output/               # generated exports and saved tests
└── src/                  # stereo core and web UI

a4/
├── images/               # own left/right stereo pair
├── output/               # rectification, disparity, and depth exports
└── src/                  # uncalibrated stereo rectification pipeline

a5/
├── data/scene/            # original and prepared reconstruction images
├── img/                   # rendered COLMAP screenshot
├── results/               # final model exports and screenshots
└── src/                   # COLMAP, DUSt3R, and visualization scripts
```

## Notes

- The scripts in `a1` work relative to the current working directory, so they
  should be started from the `a1/` folder.
- The application in `a2` loads images matching the pattern
  `a2/img/table_bottle*.jpeg` by default.
- Pressing `s` in `a2` saves an overlay image with the current geometric
  construction.
