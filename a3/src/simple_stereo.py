import argparse
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
LEFT_IMAGE = BASE_DIR / "images" / "artroom_im0.png"
RIGHT_IMAGE = BASE_DIR / "images" / "artroom_im1.png"
GROUND_TRUTH = BASE_DIR / "images" / "disp0.pfm"
CALIBRATION = BASE_DIR / "data" / "artroom_calib.npz"
OUTPUT_DIR = BASE_DIR / "output"


def format_help_text():
    return """Helpful notes:

Parameter:
  algorithm
    sgbm or bm
    sgbm usually gives more stable results, while bm is simpler and often noisier.

  block_size
    Size of the matching window.
    Smaller values preserve more detail, larger values smooth more strongly.

  uniqueness_ratio
    How distinctive a match must be.
    Higher values reject uncertain matches more aggressively.

  min_disparity
    Starting value of the disparity search range.

  num_disparities
    Width of the search range.
    Must be divisible by 16; the script adjusts the value automatically.

  speckle_window_size
    Filters small isolated error regions in the disparity map.

  speckle_range
    Tolerance of the speckle filter.

Metrics:
  MAE
    Mean Absolute Error between the computed disparity and the ground truth.

  Bad3
    Percentage of pixels with more than 3 pixels error.

Typical usage:
  .venv/bin/python a3/src/simple_stereo.py
  .venv/bin/python a3/src/simple_stereo.py --algorithm bm --block-size 15
"""


def load_calibration(path):
    calib = {}
    with np.load(path) as data:
        for key in data.files:
            value = data[key]
            if value.shape == ():
                value = float(value)
            calib[key] = value
    return calib


def load_stereo_images(left_path, right_path):
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left is None:
        raise FileNotFoundError(f"Could not load left image: {left_path}")
    if right is None:
        raise FileNotFoundError(f"Could not load right image: {right_path}")
    if left.shape != right.shape:
        raise ValueError(f"Stereo image sizes differ: {left.shape} vs {right.shape}")
    return left, right


def read_pfm(path):
    with path.open("rb") as file:
        header = file.readline().decode("ascii").rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"Not a PFM file: {path}")

        dims = file.readline().decode("ascii").strip()
        while dims.startswith("#"):
            dims = file.readline().decode("ascii").strip()
        width, height = map(int, dims.split())

        scale = float(file.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)

        channels = 3 if header == "PF" else 1
        data = np.fromfile(file, endian + "f")
        shape = (height, width, channels) if channels == 3 else (height, width)
        image = np.reshape(data, shape)
        image = np.flipud(image)
        return image.astype(np.float32), scale


def create_matcher(
    algorithm,
    calib,
    block_size,
    uniqueness_ratio=None,
    speckle_window_size=100,
    speckle_range=2,
    min_disparity=0,
    num_disparities=None,
):
    if num_disparities is None:
        num_disparities = int(np.ceil(float(calib["ndisp"]) / 16.0) * 16)
    else:
        num_disparities = int(np.ceil(float(num_disparities) / 16.0) * 16)

    if algorithm == "bm":
        matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
        matcher.setMinDisparity(min_disparity)
        matcher.setTextureThreshold(8)
        matcher.setPreFilterCap(31)
        matcher.setUniquenessRatio(10 if uniqueness_ratio is None else uniqueness_ratio)
        matcher.setSpeckleWindowSize(speckle_window_size)
        matcher.setSpeckleRange(speckle_range)
        return matcher

    channels = 1
    return cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * channels * block_size**2,
        P2=32 * channels * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=5 if uniqueness_ratio is None else uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparity(
    left,
    right,
    calib,
    algorithm="sgbm",
    block_size=5,
    uniqueness_ratio=None,
    speckle_window_size=100,
    speckle_range=2,
    min_disparity=0,
    num_disparities=None,
):
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    matcher = create_matcher(
        algorithm,
        calib,
        block_size,
        uniqueness_ratio,
        speckle_window_size,
        speckle_range,
        min_disparity,
        num_disparities,
    )
    disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity[disparity <= 0] = np.nan
    return disparity


def valid_disparity_mask(disparity, calib=None):
    mask = np.isfinite(disparity) & (disparity > 0)
    if calib is not None:
        mask &= disparity >= float(calib["vmin"])
        mask &= disparity <= float(calib["vmax"])
    return mask


def mean_absolute_error(ground_truth, prediction, calib):
    mask = valid_disparity_mask(ground_truth, calib) & np.isfinite(prediction)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs(ground_truth[mask] - prediction[mask])))


def bad_pixel_percentage(ground_truth, prediction, calib, threshold=3.0):
    mask = valid_disparity_mask(ground_truth, calib) & np.isfinite(prediction)
    if not np.any(mask):
        return np.nan
    errors = np.abs(ground_truth[mask] - prediction[mask])
    return float(np.mean(errors > threshold) * 100.0)


def evaluate_disparity(ground_truth, prediction, calib):
    mae = mean_absolute_error(ground_truth, prediction, calib)
    bad3 = bad_pixel_percentage(ground_truth, prediction, calib, threshold=3.0)
    return {
        "mae": mae,
        "bad3": bad3,
    }


def compute_depth(disparity, calib):
    focal_length = float(calib["cam0"][0, 0])
    baseline = float(calib["baseline"])
    doffs = float(calib.get("doffs", 0.0))
    denominator = disparity + doffs

    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    mask = np.isfinite(denominator) & (denominator > 0)
    depth[mask] = focal_length * baseline / denominator[mask]
    return depth


def disparity_to_points(disparity, calib):
    depth = compute_depth(disparity, calib)
    focal_length = float(calib["cam0"][0, 0])
    cx = float(calib["cam0"][0, 2])
    cy = float(calib["cam0"][1, 2])

    height, width = disparity.shape
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    points = np.empty((height, width, 3), dtype=np.float32)
    points[..., 2] = depth
    points[..., 0] = (xs.astype(np.float32) - cx) * depth / focal_length
    points[..., 1] = (ys.astype(np.float32) - cy) * depth / focal_length
    return points


def write_ply(path, points, colors, mask, max_points=500_000):
    point_values = points[mask]
    color_values = colors[mask]

    if len(point_values) > max_points:
        indices = np.linspace(0, len(point_values) - 1, max_points, dtype=np.int64)
        point_values = point_values[indices]
        color_values = color_values[indices]

    with path.open("w", encoding="ascii") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(point_values)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")

        rgb = color_values[:, ::-1]
        for point, color in zip(point_values, rgb):
            file.write(
                f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def normalize_for_display(values, mask=None, colormap=cv2.COLORMAP_PLASMA):
    if mask is None:
        mask = np.isfinite(values)

    output = np.zeros(values.shape, dtype=np.uint8)
    if np.any(mask):
        finite = values[mask]
        vmin = float(np.nanpercentile(finite, 1))
        vmax = float(np.nanpercentile(finite, 99))
        if vmax > vmin:
            scaled = (values - vmin) / (vmax - vmin)
            scaled = np.clip(scaled, 0.0, 1.0)
            output[mask] = (scaled[mask] * 255).astype(np.uint8)

    colored = cv2.applyColorMap(output, colormap)
    colored[~mask] = (0, 0, 0)
    return colored


def error_map_for_display(ground_truth, prediction, calib, max_error=12.0):
    mask = valid_disparity_mask(ground_truth, calib) & np.isfinite(prediction)
    error = np.zeros(ground_truth.shape, dtype=np.float32)
    error[mask] = np.minimum(np.abs(ground_truth[mask] - prediction[mask]), max_error)
    error_vis = np.uint8(error / max_error * 255.0)
    colored = cv2.applyColorMap(error_vis, cv2.COLORMAP_INFERNO)
    colored[~mask] = (0, 0, 0)
    return colored


def make_comparison_image(left, disparity, ground_truth, depth, calib, algorithm):
    left_small = cv2.resize(left, (480, 270), interpolation=cv2.INTER_AREA)
    disp_vis = normalize_for_display(disparity, valid_disparity_mask(disparity))
    gt_vis = normalize_for_display(ground_truth, valid_disparity_mask(ground_truth, calib))
    err_vis = error_map_for_display(ground_truth, disparity, calib)
    depth_vis = normalize_for_display(depth, np.isfinite(depth), cv2.COLORMAP_VIRIDIS)

    panels = [
        ("left", left_small),
        (f"disparity {algorithm}", cv2.resize(disp_vis, (480, 270), interpolation=cv2.INTER_AREA)),
        ("ground truth", cv2.resize(gt_vis, (480, 270), interpolation=cv2.INTER_AREA)),
        ("error", cv2.resize(err_vis, (480, 270), interpolation=cv2.INTER_AREA)),
        ("depth", cv2.resize(depth_vis, (480, 270), interpolation=cv2.INTER_AREA)),
    ]

    labeled = []
    for title, image in panels:
        panel = image.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(panel, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        labeled.append(panel)

    top = np.hstack(labeled[:3])
    bottom = np.hstack(labeled[3:] + [np.zeros_like(labeled[0])])
    return np.vstack([top, bottom])


def save_outputs(output_dir, left, disparity, ground_truth, depth, calib, algorithm):
    output_dir.mkdir(parents=True, exist_ok=True)

    disp_mask = valid_disparity_mask(disparity)
    gt_mask = valid_disparity_mask(ground_truth, calib)
    depth_mask = np.isfinite(depth)

    cv2.imwrite(str(output_dir / f"disparity_{algorithm}.png"), normalize_for_display(disparity, disp_mask))
    cv2.imwrite(str(output_dir / "ground_truth_disparity.png"), normalize_for_display(ground_truth, gt_mask))
    cv2.imwrite(str(output_dir / f"depth_{algorithm}.png"), normalize_for_display(depth, depth_mask, cv2.COLORMAP_VIRIDIS))
    cv2.imwrite(str(output_dir / f"error_{algorithm}.png"), error_map_for_display(ground_truth, disparity, calib))
    cv2.imwrite(
        str(output_dir / f"comparison_{algorithm}.png"),
        make_comparison_image(left, disparity, ground_truth, depth, calib, algorithm),
    )

    points = disparity_to_points(disparity, calib)
    cloud_mask = valid_disparity_mask(disparity, calib) & np.isfinite(points[..., 2])
    write_ply(output_dir / f"point_cloud_{algorithm}.ply", points, left, cloud_mask)


def print_report(calib, disparity, ground_truth, depth, algorithm):
    mae = mean_absolute_error(ground_truth, disparity, calib)
    bad3 = bad_pixel_percentage(ground_truth, disparity, calib, threshold=3.0)
    disp_mask = valid_disparity_mask(disparity)
    depth_mask = np.isfinite(depth)

    print("Calibration:")
    print(f"  focal length: {float(calib['cam0'][0, 0]):.2f} px")
    print(f"  baseline:     {float(calib['baseline']):.2f}")
    print(f"  ndisp:        {float(calib['ndisp']):.0f}")
    print(f"  valid disp:   {float(calib['vmin']):.1f}..{float(calib['vmax']):.1f} px")
    print()
    print(f"Stereo matcher: {algorithm}")
    print(f"  disparity:    {np.nanmin(disparity[disp_mask]):.2f}..{np.nanmax(disparity[disp_mask]):.2f} px")
    print(f"  depth:        {np.nanmin(depth[depth_mask]):.2f}..{np.nanmax(depth[depth_mask]):.2f}")
    print(f"  MAE:          {mae:.2f} px")
    print(f"  bad pixels:   {bad3:.2f}% (> 3 px)")


def run_single(
    algorithm,
    block_size,
    output_dir,
    save=True,
    uniqueness_ratio=5,
    speckle_window_size=100,
    speckle_range=2,
    min_disparity=0,
    num_disparities=None,
):
    calib = load_calibration(CALIBRATION)
    left, right = load_stereo_images(LEFT_IMAGE, RIGHT_IMAGE)
    ground_truth, _ = read_pfm(GROUND_TRUTH)

    disparity = compute_disparity(
        left,
        right,
        calib,
        algorithm,
        block_size,
        uniqueness_ratio,
        speckle_window_size,
        speckle_range,
        min_disparity,
        num_disparities,
    )
    depth = compute_depth(disparity, calib)

    if save:
        save_outputs(output_dir, left, disparity, ground_truth, depth, calib, algorithm)

    return {
        "calib": calib,
        "left": left,
        "ground_truth": ground_truth,
        "disparity": disparity,
        "depth": depth,
        "metrics": evaluate_disparity(ground_truth, disparity, calib),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Workshop 03: simple stereo disparity, depth, and point cloud.",
        epilog=format_help_text(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--algorithm", choices=("sgbm", "bm"), default="sgbm")
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--uniqueness-ratio", type=int, default=5)
    parser.add_argument("--speckle-window-size", type=int, default=100)
    parser.add_argument("--speckle-range", type=int, default=2)
    parser.add_argument("--min-disparity", type=int, default=0)
    parser.add_argument("--num-disparities", type=int, default=176)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.block_size % 2 == 0 or args.block_size < 5:
        raise ValueError("--block-size must be an odd integer >= 5")

    result = run_single(
        args.algorithm,
        args.block_size,
        args.output_dir,
        save=True,
        uniqueness_ratio=args.uniqueness_ratio,
        speckle_window_size=args.speckle_window_size,
        speckle_range=args.speckle_range,
        min_disparity=args.min_disparity,
        num_disparities=args.num_disparities,
    )
    print_report(result["calib"], result["disparity"], result["ground_truth"], result["depth"], args.algorithm)
    print(f"\nSaved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
