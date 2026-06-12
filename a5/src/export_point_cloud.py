import argparse
import json
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BASE_DIR.parent
DEFAULT_COLOR = REPO_DIR / "a4" / "output" / "test_nd256_b7_u10" / "02_rectified_left.png"
DEFAULT_DEPTH = REPO_DIR / "a4" / "output" / "test_nd256_b7_u10" / "08_depth.npy"
DEFAULT_OUTPUT = BASE_DIR / "output" / "final_reconstruction.ply"


def load_inputs(color_path, depth_path):
    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(f"Could not read color image: {color_path}")

    depth = np.load(depth_path).astype(np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got shape {depth.shape}")
    if color.shape[:2] != depth.shape:
        raise ValueError(f"Color/depth size mismatch: {color.shape[:2]} vs {depth.shape}")
    return color, depth


def depth_to_points(depth, focal_px):
    height, width = depth.shape
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    points = np.empty((height, width, 3), dtype=np.float32)
    points[..., 0] = (xs - cx) * depth / focal_px
    points[..., 1] = -(ys - cy) * depth / focal_px
    points[..., 2] = depth
    return points


def select_points(points, colors, depth, min_depth, max_depth, max_points):
    mask = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    selected_points = points[mask]
    selected_colors = colors[mask, ::-1]

    if selected_points.size == 0:
        raise RuntimeError("No points remain after depth filtering.")

    if len(selected_points) > max_points:
        indices = np.linspace(0, len(selected_points) - 1, max_points, dtype=np.int64)
        selected_points = selected_points[indices]
        selected_colors = selected_colors[indices]
    return selected_points, selected_colors, int(np.count_nonzero(mask))


def write_ascii_ply(path, points, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, colors):
            file.write(
                f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Export the A5 stereo result as a colored PLY point cloud.")
    parser.add_argument("--color", type=Path, default=DEFAULT_COLOR)
    parser.add_argument("--depth", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--focal-px", type=float, default=1296.296296)
    parser.add_argument("--min-depth", type=float, default=0.80)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--max-points", type=int, default=150_000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.focal_px <= 0:
        raise ValueError("--focal-px must be positive")
    if args.min_depth >= args.max_depth:
        raise ValueError("--min-depth must be smaller than --max-depth")
    if args.max_points <= 0:
        raise ValueError("--max-points must be positive")

    color, depth = load_inputs(args.color, args.depth)
    points = depth_to_points(depth, args.focal_px)
    selected_points, selected_colors, valid_before_sampling = select_points(
        points,
        color,
        depth,
        args.min_depth,
        args.max_depth,
        args.max_points,
    )
    write_ascii_ply(args.output, selected_points, selected_colors)

    metadata = {
        "source_color": str(args.color),
        "source_depth": str(args.depth),
        "focal_px": args.focal_px,
        "min_depth_m": args.min_depth,
        "max_depth_m": args.max_depth,
        "valid_points_before_sampling": valid_before_sampling,
        "exported_points": len(selected_points),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Exported {len(selected_points):,} colored points to {args.output}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
