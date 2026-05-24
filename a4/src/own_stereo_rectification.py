import argparse
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LEFT = BASE_DIR / "images" / "my_pair_left.png"
DEFAULT_RIGHT = BASE_DIR / "images" / "my_pair_right.png"
DEFAULT_OUTPUT = BASE_DIR / "output"


def load_camera_calibration(path):
    with np.load(path, allow_pickle=True) as data:
        if "camera_matrix" in data:
            camera_matrix = data["camera_matrix"].astype(np.float64)
        elif "cam0" in data:
            camera_matrix = data["cam0"].astype(np.float64)
        else:
            raise KeyError(f"Could not find 'camera_matrix' or 'cam0' in calibration file: {path}")

        if "dist_coeffs" in data:
            dist_coeffs = data["dist_coeffs"].astype(np.float64)
        else:
            dist_coeffs = np.zeros((1, 5), dtype=np.float64)

    return camera_matrix, dist_coeffs


def scale_camera_matrix(camera_matrix, scale):
    scaled = camera_matrix.copy()
    scaled[0, 0] *= scale
    scaled[1, 1] *= scale
    scaled[0, 2] *= scale
    scaled[1, 2] *= scale
    return scaled


def load_images(left_path, right_path):
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)

    if left is None:
        raise FileNotFoundError(f"Could not read left image: {left_path}")
    if right is None:
        raise FileNotFoundError(f"Could not read right image: {right_path}")
    if left.shape != right.shape:
        raise ValueError(f"Left and right image must have the same size: {left.shape} vs {right.shape}")

    return left, right


def create_feature_detector(name, max_features):
    if name == "auto":
        name = "sift" if hasattr(cv2, "SIFT_create") else "orb"

    if name == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("SIFT is not available in this OpenCV build. Use --detector orb.")
        return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2
    if name == "akaze":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    if name == "orb":
        return cv2.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING

    raise ValueError(f"Unknown detector '{name}'. Choose auto, sift, akaze, or orb.")


def prepare_gray_for_features(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def resize_for_processing(left, right, max_size):
    if max_size is None or max_size <= 0:
        return left, right, 1.0

    height, width = left.shape[:2]
    longest = max(height, width)
    if longest <= max_size:
        return left, right, 1.0

    scale = max_size / float(longest)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    resized_left = cv2.resize(left, new_size, interpolation=cv2.INTER_AREA)
    resized_right = cv2.resize(right, new_size, interpolation=cv2.INTER_AREA)
    return resized_left, resized_right, scale


def detect_and_match(gray_left, gray_right, max_features=12000, ratio=0.72, detector_name="auto"):
    detector, norm = create_feature_detector(detector_name, max_features)
    feature_left = prepare_gray_for_features(gray_left)
    feature_right = prepare_gray_for_features(gray_right)
    keypoints_left, descriptors_left = detector.detectAndCompute(feature_left, None)
    keypoints_right, descriptors_right = detector.detectAndCompute(feature_right, None)

    if descriptors_left is None or descriptors_right is None:
        raise RuntimeError("Could not compute descriptors in both images.")

    matcher = cv2.BFMatcher(norm, crossCheck=False)
    raw_matches = matcher.knnMatch(descriptors_left, descriptors_right, k=2)

    matches = []
    for candidate in raw_matches:
        if len(candidate) != 2:
            continue
        first, second = candidate
        if first.distance < ratio * second.distance:
            matches.append(first)

    if len(matches) < 8:
        raise RuntimeError(f"Need at least 8 matches for F estimation, got {len(matches)}.")

    matches = sorted(matches, key=lambda match: match.distance)
    points_left = np.float32([keypoints_left[match.queryIdx].pt for match in matches])
    points_right = np.float32([keypoints_right[match.trainIdx].pt for match in matches])
    return keypoints_left, keypoints_right, matches, points_left, points_right


def estimate_fundamental_matrix(points_left, points_right, ransac_threshold=1.0, confidence=0.99):
    fundamental, mask = cv2.findFundamentalMat(
        points_left,
        points_right,
        cv2.FM_RANSAC,
        ransac_threshold,
        confidence,
    )

    if fundamental is None or fundamental.shape != (3, 3) or mask is None:
        raise RuntimeError("Could not estimate a valid fundamental matrix.")

    inlier_mask = mask.ravel().astype(bool)
    inliers_left = points_left[inlier_mask]
    inliers_right = points_right[inlier_mask]

    if len(inliers_left) < 8:
        raise RuntimeError(f"RANSAC kept too few inliers: {len(inliers_left)}.")

    return fundamental, inlier_mask, inliers_left, inliers_right


def rectify_uncalibrated(left, right, gray_left, inliers_left, inliers_right, fundamental):
    height, width = gray_left.shape
    ok, homography_left, homography_right = cv2.stereoRectifyUncalibrated(
        inliers_left,
        inliers_right,
        fundamental,
        imgSize=(width, height),
    )

    if not ok:
        raise RuntimeError("Rectification failed.")

    rect_left = cv2.warpPerspective(left, homography_left, (width, height))
    rect_right = cv2.warpPerspective(right, homography_right, (width, height))
    return homography_left, homography_right, rect_left, rect_right


def rectify_calibrated(left, right, points_left, points_right, camera_matrix, dist_coeffs):
    height, width = left.shape[:2]
    essential, mask = cv2.findEssentialMat(
        points_left,
        points_right,
        camera_matrix,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )

    if essential is None or mask is None:
        raise RuntimeError("Could not estimate a valid essential matrix.")

    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential,
        points_left,
        points_right,
        camera_matrix,
        mask=mask,
    )

    rect_left_rot, rect_right_rot, proj_left, proj_right, _, _, _ = cv2.stereoRectify(
        camera_matrix,
        dist_coeffs,
        camera_matrix,
        dist_coeffs,
        (width, height),
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        rect_left_rot,
        proj_left,
        (width, height),
        cv2.CV_32FC1,
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        rect_right_rot,
        proj_right,
        (width, height),
        cv2.CV_32FC1,
    )

    rect_left = cv2.remap(left, map_left_x, map_left_y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right, map_right_x, map_right_y, cv2.INTER_LINEAR)
    inlier_count = int(np.count_nonzero(pose_mask))
    return rect_left, rect_right, inlier_count


def maybe_swap_rectified_pair(rect_left, rect_right, rectified_inliers_left, rectified_inliers_right):
    median_disparity = float(np.median(rectified_inliers_left[:, 0] - rectified_inliers_right[:, 0]))
    if median_disparity >= 0:
        return rect_left, rect_right, False, median_disparity
    return rect_right, rect_left, True, median_disparity


def compute_vertical_error(points_left, points_right):
    if len(points_left) == 0:
        return float("nan"), float("nan")
    differences = np.abs(points_left[:, 1] - points_right[:, 1])
    return float(np.mean(differences)), float(np.median(differences))


def transform_points(points, homography):
    transformed = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography)
    return transformed.reshape(-1, 2)


def draw_match_visualization(left, keypoints_left, right, keypoints_right, matches, inlier_mask, max_matches=80):
    inlier_matches = [match for match, is_inlier in zip(matches, inlier_mask) if is_inlier]
    selected = inlier_matches[:max_matches]
    return cv2.drawMatches(
        left,
        keypoints_left,
        right,
        keypoints_right,
        selected,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def draw_rectification_check(rect_left, rect_right, line_step=40):
    combined = np.hstack([rect_left, rect_right])
    height, width = rect_left.shape[:2]

    for y in range(line_step, height, line_step):
        color = (0, 255, 255) if (y // line_step) % 2 else (255, 180, 0)
        cv2.line(combined, (0, y), (2 * width - 1, y), color, 1, cv2.LINE_AA)

    cv2.line(combined, (width, 0), (width, height - 1), (255, 255, 255), 1, cv2.LINE_AA)
    return combined


def create_sgbm(num_disparities=176, block_size=5, uniqueness_ratio=10):
    num_disparities = int(np.ceil(num_disparities / 16.0) * 16)
    if block_size % 2 == 0 or block_size < 3:
        raise ValueError("--block-size must be an odd integer >= 3")

    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size**2,
        P2=32 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparity(rect_left, rect_right, num_disparities, block_size, uniqueness_ratio):
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
    matcher = create_sgbm(num_disparities, block_size, uniqueness_ratio)
    disparity = matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
    disparity[disparity <= 0] = np.nan
    return disparity


def compute_depth(disparity, focal_px=None, baseline=None):
    valid = np.isfinite(disparity) & (disparity > 0)
    depth = np.full_like(disparity, np.nan, dtype=np.float32)

    if focal_px is not None and baseline is not None:
        depth[valid] = (float(focal_px) * float(baseline)) / disparity[valid]
        return depth, "metric"

    depth[valid] = 1.0 / disparity[valid]
    return depth, "relative"


def compute_value_stats(values):
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return None

    return {
        "min": float(np.nanmin(valid)),
        "max": float(np.nanmax(valid)),
        "mean": float(np.nanmean(valid)),
        "median": float(np.nanmedian(valid)),
        "p05": float(np.nanpercentile(valid, 5)),
        "p95": float(np.nanpercentile(valid, 95)),
        "valid_pixels": int(valid.size),
        "total_pixels": int(values.size),
    }


def normalize_for_display(values, colormap=cv2.COLORMAP_PLASMA):
    mask = np.isfinite(values)
    output = np.zeros(values.shape, dtype=np.uint8)

    if np.any(mask):
        vmin = float(np.nanpercentile(values[mask], 1))
        vmax = float(np.nanpercentile(values[mask], 99))
        if vmax > vmin:
            scaled = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
            output[mask] = (scaled[mask] * 255).astype(np.uint8)

    colored = cv2.applyColorMap(output, colormap)
    colored[~mask] = (0, 0, 0)
    return colored


def save_outputs(output_dir, outputs):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, image in outputs.items():
        cv2.imwrite(str(output_dir / name), image)


def save_float_array(output_dir, name, values):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / name, values.astype(np.float32))


def resolve_depth_unit(depth_mode, baseline_unit):
    if depth_mode != "metric":
        return "relative 1/px"
    return baseline_unit


def save_depth_stats(output_dir, depth, depth_mode, focal_px, baseline, baseline_unit):
    stats = compute_value_stats(depth)
    unit = resolve_depth_unit(depth_mode, baseline_unit)
    lines = [
        f"depth mode: {depth_mode}",
        f"unit: {unit}",
        f"focal length px: {focal_px:.6f}" if focal_px is not None else "focal length px: n/a",
        f"baseline: {baseline:.6f} {baseline_unit}" if baseline is not None else "baseline: n/a",
    ]

    if stats is None:
        lines.append("valid pixels: 0")
    else:
        lines.extend(
            [
                f"valid pixels: {stats['valid_pixels']} / {stats['total_pixels']}",
                f"min: {stats['min']:.6f} {unit}",
                f"p05: {stats['p05']:.6f} {unit}",
                f"median: {stats['median']:.6f} {unit}",
                f"mean: {stats['mean']:.6f} {unit}",
                f"p95: {stats['p95']:.6f} {unit}",
                f"max: {stats['max']:.6f} {unit}",
            ]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "09_depth_stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Workshop 04: own stereo rectification and disparity.")
    parser.add_argument("--left", type=Path, default=DEFAULT_LEFT)
    parser.add_argument("--right", type=Path, default=DEFAULT_RIGHT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--detector", choices=["auto", "sift", "akaze", "orb"], default="auto")
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--ratio", type=float, default=0.72)
    parser.add_argument("--ransac-threshold", type=float, default=1.5)
    parser.add_argument("--max-image-size", type=int, default=1400)
    parser.add_argument("--num-disparities", type=int, default=176)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--uniqueness-ratio", type=int, default=10)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--focal-px", type=float, default=None)
    parser.add_argument("--baseline", type=float, default=None)
    parser.add_argument("--baseline-unit", default="m")
    return parser.parse_args()


def main():
    args = parse_args()
    left, right = load_images(args.left, args.right)
    left, right, scale = resize_for_processing(left, right, args.max_image_size)
    camera_matrix = None
    dist_coeffs = None
    if args.calibration is not None:
        camera_matrix, dist_coeffs = load_camera_calibration(args.calibration)
        camera_matrix = scale_camera_matrix(camera_matrix, scale)
    focal_px = args.focal_px
    if focal_px is None and camera_matrix is not None:
        focal_px = float(camera_matrix[0, 0])
    elif focal_px is not None:
        focal_px *= scale

    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    keypoints_left, keypoints_right, matches, points_left, points_right = detect_and_match(
        gray_left,
        gray_right,
        args.max_features,
        args.ratio,
        args.detector,
    )
    fundamental, inlier_mask, inliers_left, inliers_right = estimate_fundamental_matrix(
        points_left,
        points_right,
        args.ransac_threshold,
    )

    before_mean, before_median = compute_vertical_error(inliers_left, inliers_right)

    pose_inliers = None
    rectification_mode = "uncalibrated"
    if camera_matrix is not None and dist_coeffs is not None:
        rect_left, rect_right, pose_inliers = rectify_calibrated(
            left,
            right,
            points_left,
            points_right,
            camera_matrix,
            dist_coeffs,
        )
        rect_left_for_disparity = rect_left
        rect_right_for_disparity = rect_right
        swapped_for_disparity = False
        median_rectified_disparity = float("nan")
        after_mean = float("nan")
        after_median = float("nan")
        rectification_mode = "calibrated"
    else:
        homography_left, homography_right, rect_left, rect_right = rectify_uncalibrated(
            left,
            right,
            gray_left,
            inliers_left,
            inliers_right,
            fundamental,
        )

        rectified_inliers_left = transform_points(inliers_left, homography_left)
        rectified_inliers_right = transform_points(inliers_right, homography_right)
        after_mean, after_median = compute_vertical_error(rectified_inliers_left, rectified_inliers_right)
        rect_left_for_disparity, rect_right_for_disparity, swapped_for_disparity, median_rectified_disparity = maybe_swap_rectified_pair(
            rect_left,
            rect_right,
            rectified_inliers_left,
            rectified_inliers_right,
        )

    disparity = compute_disparity(
        rect_left_for_disparity,
        rect_right_for_disparity,
        args.num_disparities,
        args.block_size,
        args.uniqueness_ratio,
    )
    depth, depth_mode = compute_depth(disparity, focal_px, args.baseline)

    save_outputs(
        args.output_dir,
        {
            "01_matches_inliers.png": draw_match_visualization(
                left,
                keypoints_left,
                right,
                keypoints_right,
                matches,
                inlier_mask,
            ),
            "02_rectified_left.png": rect_left,
            "03_rectified_right.png": rect_right,
            "04_rectification_check.png": draw_rectification_check(rect_left, rect_right),
            "05_disparity.png": normalize_for_display(disparity),
            "06_depth.png": normalize_for_display(depth, cv2.COLORMAP_VIRIDIS),
        },
    )
    save_float_array(args.output_dir, "07_disparity.npy", disparity)
    save_float_array(args.output_dir, "08_depth.npy", depth)
    save_depth_stats(args.output_dir, depth, depth_mode, focal_px, args.baseline, args.baseline_unit)

    depth_stats = compute_value_stats(depth)
    depth_unit = resolve_depth_unit(depth_mode, args.baseline_unit)

    print("Workshop 04 result")
    print(f"  left image:             {args.left}")
    print(f"  right image:            {args.right}")
    print(f"  processing scale:       {scale:.3f}")
    print(f"  rectification mode:     {rectification_mode}")
    print(f"  detector:               {args.detector}")
    print(f"  keypoints left/right:   {len(keypoints_left)} / {len(keypoints_right)}")
    print(f"  ratio-test matches:     {len(matches)}")
    print(f"  RANSAC inliers:         {len(inliers_left)}")
    print(f"  vertical error before:  mean {before_mean:.2f}px, median {before_median:.2f}px")
    if np.isfinite(after_mean) and np.isfinite(after_median):
        print(f"  vertical error after:   mean {after_mean:.2f}px, median {after_median:.2f}px")
    if np.isfinite(median_rectified_disparity):
        print(f"  median rect disparity:  {median_rectified_disparity:.2f}px")
    print(f"  swapped for disparity:  {swapped_for_disparity}")
    if pose_inliers is not None:
        print(f"  pose inliers:           {pose_inliers}")
    print(f"  depth mode:             {depth_mode}")
    if focal_px is not None:
        print(f"  focal length:           {focal_px:.2f}px")
    if args.baseline is not None:
        print(f"  baseline:               {args.baseline:.4f} {args.baseline_unit}")
    if depth_stats is not None:
        print(
            "  depth median/mean:      "
            f"{depth_stats['median']:.4f} / {depth_stats['mean']:.4f} {depth_unit}"
        )
    print(f"  output dir:             {args.output_dir}")
    print()
    print("Fundamental matrix:")
    print(fundamental)


if __name__ == "__main__":
    main()
