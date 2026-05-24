# 3DCV Workshop 04 - Rectify Your Own Stereo Pair

## Workshop Overview

In this workshop, you will use your own two images of the same scene to build a small stereo pipeline. You will:

1. capture or load your own left/right image pair,
2. estimate the fundamental matrix from feature matches,
3. rectify the images so that corresponding points lie on the same image rows,
4. reuse your disparity code from Workshop 03 on the rectified pair,
5. compute a depth map from disparity.

This workshop connects feature matching, epipolar geometry, rectification, and stereo reconstruction.

Important: Do not copy a complete ready-made solution. Implement step by step using the short hints below.

## Preparation

Before you start, create a stereo pair yourself:

- Use the same camera for both images.
- Keep the camera rotation small between the two shots.
- Move the camera mostly sideways between the two shots.
- Avoid moving objects, and very shiny or textureless surfaces.
- Make sure the scene contains enough texture for feature matching.

Suggested file names:

- `images/my_pair_left.png`
- `images/my_pair_right.png`

Note that you need to find out which image is left and which is right, because you will not have calibration data to check the sign of the disparity. If you get negative disparities, swap the order of the images in your code.


### Optional: Metric Depth

For disparity in pixel units, rectification is enough. For metric depth in meters or centimeters, you additionally need:

- a focal length in pixels, for example from Workshop 01 calibration, and
- the baseline $B$ between the two camera positions, measured manually.

Without these values, you can still compute a relative depth map, but not an absolute metric one.

In the implementation, metric depth is computed with the pinhole-camera stereo formula:

$$
Z = \frac{f \cdot B}{d}
$$

where:

- $Z$ is the depth,
- $f$ is the focal length in pixels,
- $B$ is the measured baseline between the two camera positions,
- $d$ is the disparity in pixels.

The unit of $Z$ follows the unit of $B$. For example, if the baseline is entered in meters, the depth values are in meters. If no focal length and baseline are provided, the implementation stores relative depth as:

$$
Z_{relative} = \frac{1}{d}
$$

Example with manually entered focal length and baseline:

```bash
.venv/bin/python a4/src/own_stereo_rectification.py \
  --left a4/images/IMG_6405.jpg \
  --right a4/images/IMG_6406.jpg \
  --baseline 0.08 \
  --baseline-unit m \
  --focal-px 1200
```

Example with focal length loaded from a calibration file:

```bash
.venv/bin/python a4/src/own_stereo_rectification.py \
  --left a4/images/IMG_6405.jpg \
  --right a4/images/IMG_6406.jpg \
  --baseline 0.08 \
  --baseline-unit m \
  --calibration a1/calibration.npz
```

The script saves both visualization images and numeric arrays. The metric or relative depth values are written to `08_depth.npy`, and a short summary is written to `09_depth_stats.txt`. A colored point cloud can be exported optionally, for example by reusing the export code from Workshop 03.

---

## Step 1 - Load your images and optional calibration parameters

### Task

Load your left/right images. If you already calibrated your camera in Workshop 01, also load the intrinsic matrix.

### Minimal hint

```python
from pathlib import Path
import cv2
import numpy as np

left = cv2.imread("images/my_pair_left.png", cv2.IMREAD_COLOR)
right = cv2.imread("images/my_pair_right.png", cv2.IMREAD_COLOR)

if left is None or right is None:
			raise FileNotFoundError("Could not read both stereo images.")

if left.shape != right.shape:
			raise ValueError("Left and right image must have the same size.")

gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
```

If you have calibration data from Workshop 01:

```python
with np.load("calibration.npz") as data:
	K = data["camera_matrix"]
```

### Checkpoint

- Both images load without errors.
- The images have the same resolution.
- If available, your intrinsic matrix has plausible values.

---

## Step 2 - Detect and match features

### Task A - Detection

Detect distinctive keypoints in both images and match them. Start with [ORB](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html) because it is developed by the "OpenCV Labs". You should check if you get better results with [SIFT](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html) later.

### Task B - Matching

Match the keypoints using a brute-force matcher. Start with `crossCheck=True` for simplicity, and then try `knnMatch()` with Lowe's ratio test for better results. Check this [OpenCV tutorial](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html) for more information.

### Visualization hint

Use `cv2.drawMatches()` on the best 50 to 100 matches and inspect whether the correspondences look plausible.

### Checkpoint

- You detect enough keypoints in both images.
- The best matches mostly connect the same physical structures.
- Obvious outliers are still present, which is expected before RANSAC.

### Challenge

Try `knnMatch()` with Lowe's ratio test instead of `crossCheck=True` and compare the quality of the inlier set.

---

## Step 3 - Estimate the fundamental matrix

### Task

Estimate the fundamental matrix $F$ from your matched points using RANSAC. This filters out bad correspondences automatically.

### Minimal hint

```python
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
if F is None:
			raise RuntimeError("Could not estimate a fundamental matrix.")

inlier_mask = mask.ravel().astype(bool)
inliers_left = points_left[inlier_mask]
inliers_right = points_right[inlier_mask]
```

### Why this works

The fundamental matrix encodes the epipolar geometry between two uncalibrated views:

$$
x_2^T F x_1 = 0
$$

for corresponding image points $x_1$ and $x_2$ in homogeneous coordinates.

### Checkpoint

- `F` is a `3x3` matrix.
- You keep a reasonable number of inlier matches after RANSAC.
- Repeated structures should still be handled robustly if enough correct matches exist.

---

## Step 4 - Rectify the images from the fundamental matrix

### Task

Use the inlier correspondences and the fundamental matrix to compute two homographies that rectify the images.

### Minimal hint

```python
h, w = gray_left.shape

ok, H1, H2 = cv2.stereoRectifyUncalibrated(
			inliers_left,
			inliers_right,
			F,
			imgSize=(w, h),
)
if not ok:
			raise RuntimeError("Rectification failed.")

rect_left = cv2.warpPerspective(left, H1, (w, h))
rect_right = cv2.warpPerspective(right, H2, (w, h))
```

### Visualization hint

Stack the rectified images side by side and draw a few horizontal guide lines. If rectification worked, corresponding points should appear on the same horizontal rows.

### Checkpoint

- Rectified images are produced without errors.
- Most corresponding structures align horizontally.
- Vertical disparity is strongly reduced.

### Challenge

Draw epipolar lines before and after rectification. Explain why the post-rectification epipolar lines become horizontal.

---

## Step 5 - Reuse workshop 03 for disparity

### Task

Now reuse your disparity code from Workshop 03, but run it on the rectified grayscale images instead of the provided `artroom` pair.

### Minimal hint

#### Parameters for StereoSGBM (you can experiment with these):

```python
# OpenCV requires numDisparities to be divisible by 16
num_disp = int(np.ceil(NDISP / 16.0) * 16)  # 176 for NDISP=170
block_size = 5
matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 1 * block_size**2,
    P2=32 * 1 * block_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    preFilterCap=31,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)
```

Do not forget to divide the raw disparity by 16.0 to get the actual disparity in pixel units.

```python
# After computing the raw disparity, you can also apply some post-processing to filter out invalid values:
disparity = matcher.compute(left, right).astype(np.float32) / 16.0
```

### Checkpoint

- You get a visible disparity map on the rectified pair.
- Nearby objects usually have larger disparity than distant ones.

---

## Step 6 - Compute Depth Map

### Task

Convert disparity to depth. If you know focal length $f$ and baseline $B$, compute metric depth:

$$
Z = \frac{f \cdot B}{d}
$$

If you do not know $f$ and $B$, compute only a relative depth map by inverting the disparity.

### Minimal hint

```python
depth = np.full_like(disparity, np.nan, dtype=np.float32)
valid = np.isfinite(disparity) & (disparity > 0)

focal_px = K[0, 0]
baseline_m = 0.08  # example: 8 cm measured by hand
depth[valid] = (focal_px * baseline_m) / disparity[valid]
```

For relative depth only:

```python
relative_depth = np.full_like(disparity, np.nan, dtype=np.float32)
relative_depth[valid] = 1.0 / disparity[valid]
```

### Checkpoint

- Invalid disparities do not cause division-by-zero errors.
- The depth map is plausible: close objects are smaller in $Z$ than far objects.
- If you use calibrated intrinsics and measured baseline, the values are in a plausible metric range.

---

## Step 7 - Optional Evaluation and Export

### Task

Since you created your own stereo pair, you probably do not have ground truth disparity. Instead, evaluate your pipeline qualitatively:

- inspect the match quality,
- inspect inliers after RANSAC,
- inspect horizontal alignment after rectification,
- inspect whether depth ordering in the scene is plausible.

If you want, reuse the point cloud export from Workshop 03 on your rectified pair.

---

## Suggested Program Structure

You can split the work into small functions like this:

```python
def load_images(...):
	...

def match_features(...):
	...

def estimate_fundamental(...):
	...

def rectify_pair(...):
	...

def compute_disparity(...):
	...

def disparity_to_depth(...):
	...
```

This makes it easier to replace one stage without breaking the rest of the pipeline.

---

## Submission

Submit your code as Gitlab repository. Include:

- your two input images,
- your rectified image pair,
- one disparity visualization,
- one depth visualization,
- a short README that explains your capture setup,
- whether your depth is metric or only relative.

### Grading Criteria

- Correct estimation of the fundamental matrix and rectification quality (35%)
- Quality of the disparity and depth results (35%)
- Code quality and documentation (20%)
- Experimentation and discussion of failure cases (10%)
