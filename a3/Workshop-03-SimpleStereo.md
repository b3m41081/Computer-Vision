# 3DCV Workshop 03 - Simple Stereo

## Workshop Overview

In this workshop, you will estimate the disparity and depth of a scene from a stereo image pair using OpenCV's stereo matching algorithms. The ground truth is provided so that you can evaluate your results.

We use a [data set from the Middlebury Vision group](https://vision.middlebury.edu/stereo/data/scenes2021/). You are given two stereo images of an art room scene:

- `images/artroom_im0.png`
- `images/artroom_im1.png`

as well as the corresponding ground truth disparity map and calibration data:

- `images/disp0.pfm`
- `data/artroom_calib.npz`

Your goal is to compute a disparity map from the stereo pair and evaluate it against the ground truth.

## What You Build

Create a Python + OpenCV pipeline that:

1. Lets you load the stereo images and calibration data.
2. Computes the disparity map using OpenCV's stereo matching algorithms (e.g., StereoBM or StereoSGBM).
3. Evaluates the computed disparity map against the ground truth using appropriate metrics (e.g., mean absolute error, percentage of bad pixels).
4. Computes the depth map from the disparity map using the calibration data.
5. Exports the computed disparity map as a point cloud in PLY format.

Important: Do not copy a complete ready-made solution. Implement step by step using the short hints below.

---

## Step 1 - Load Stereo Images and Calibration Data

### Task

Implement a function to load the stereo images and the calibration data. The calibration data includes the focal length and baseline, which are necessary for depth computation.

### Minimal hint

```python
def load_calibration(path):
	calib = {}
	with np.load(path) as data:
		for key in data.files:
			calib[key] = data[key]
        return calib
```

### Checkpoint

- The stereo images and calibration data are loaded successfully without errors.
- The values of focal length and baseline are plausible.

---

## Step 2 - Compute Disparity Map

### Task

Use OpenCV's stereo matching algorithms to compute the disparity map from the stereo image pair. You can start with a simple block matching algorithm ([StereoBM](https://docs.opencv.org/4.x/d9/dba/classcv_1_1StereoBM.html)) and then try a more advanced one ([StereoSGBM](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html)) for better results. You can check the OpenCV documentation for parameters and usage details as well as the official [OpenCV sample for stereo matching](https://github.com/opencv/opencv/blob/4.x/samples/python/stereo_match.py).

### Minimal hint

Get inspired by the following code snippet to create a stereo matcher and compute the disparity map:

```python
# OpenCV requires numDisparities to be divisible by 16
num_disp = int(np.ceil(NDISP / 16.0) * 16)  # 176 for NDISP=170 as in the provided calibration data
matcher = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = matcher.compute(left_image, right_image).astype(np.float32) / 16.0
```

### Visualization hint

In order to visualize the disparity map, you can normalize it to the range [0, 255] and convert it to uint8 and apply a colormap:

```python
disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disp_vis = np.uint8(disp_vis)
disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_PLASMA)
cv2.imshow("Disparity", disp_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Checkpoint

The disparity map is computed and visualized successfully. The disparity values are in a plausible range (e.g., non-negative and not exceeding the maximum disparity). It looks similar to the ground truth disparity map.

---

## Step 3 - Evaluate Disparity Map

### Task

Evaluate the computed disparity map against the ground truth using appropriate metrics. Common metrics include mean absolute error (MAE) and percentage of bad pixels (where the disparity is outside the range given in the calibration data).

### Hint

You can compute the mean absolute error as follows:

```python
def mean_absolute_error(gt, pred):
	mask = ~np.isnan(gt)  # Only consider valid pixels in the ground truth
	return np.mean(np.abs(gt[mask] - pred[mask]))
```

You can use the following method to read the ground truth disparity map from [PFM file](https://www.pauldebevec.com/Research/HDR/PFM/):

```python
def read_pfm(path: Path):
    """Read PFM file, return (image, scale)."""
    with path.open("rb") as f:
        header = f.readline().decode("ascii").rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"Not a PFM file: {path}")

        dims = f.readline().decode("ascii").strip()
        while dims.startswith("#"):
            dims = f.readline().decode("ascii").strip()
        w, h = map(int, dims.split())

        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        scale = abs(scale) # scale is typically used to convert disparity values to real-world units, but we can ignore it for now since we will compute depth from disparity using the calibration data.

        channels = 3 if header == "PF" else 1
        data = np.fromfile(f, endian + "f")
        shape = (h, w, channels) if channels == 3 else (h, w)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data.astype(np.float32), scale
```

### Checkpoint

You can visualize disparity map and the groud truth map side by side and compute the mean absolute error. The error should be in a reasonable range (e.g., less than 10 pixels on average).

---

## Step 4 - Compute Depth Map

### Task

Convert the computed disparity map to a depth map using the calibration data (focal length and baseline). The formula for depth from disparity is:

$$
Z = \frac{f \cdot B}{d}
$$

### Hint

You have to take care of invalid disparity values (e.g., zero or negative) to avoid division by zero. You can set the depth to NaN for those pixels.

```python
disparity[disparity <= 0] = np.nan  # Set invalid disparities to NaN
```

### Checkpoint

The depth map is computed successfully. The depth values are in a plausible range (e.g., positive and not excessively large). You can visualize the depth map using a similar approach as for the disparity map.

---

## Step 5 - Export Point Cloud

### Task

Use the computed disparity map and the calibration data to reproject the points into 3D space and export them as a point cloud in PLY format. You can use OpenCV's [`cv2.reprojectImageTo3D()`](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga1bc1152bd57d63bc524204f21fde6e02) function for this purpose. Also check the [OpenCV sample for stereo matching](https://github.com/opencv/opencv/blob/4.x/samples/python/stereo_match.py) for an example of how to export a point cloud.

### Hint

If you visualize the point cloud in a 3D viewer (e.g., MeshLab), you should see a 3D representation of the scene. You can also color the points using the left image for better visualization. Take care to filter out points with invalid disparity values before exporting. You can find the valid range of disparities from the calibration data (e.g., `calib["vmin"]` and `calib["vmax"]`) and only export points with disparities within this range.

---

## Optional Challenge - Experiment with Different Stereo Matching Algorithms

Try different stereo matching algorithms (e.g., StereoSGBM) and compare the results in terms of disparity quality and evaluation metrics. You can also experiment with different parameters of the algorithms to see how they affect the results.

## Submission

Submit your code as Gitlab repository. Make sure to include a README file that explains how to run your code and any dependencies required. You can also include visualizations of the disparity map, depth map, and point cloud in your README for better presentation. There will be a small prize for the best solution in terms of accuracy.

### Grading Criteria

- Correctness of the disparity and depth computation (50%)
- Quality of the evaluation and analysis (20%)
- Code quality and documentation (readme) (20%)
- Creativity and experimentation with different algorithms (10%)

