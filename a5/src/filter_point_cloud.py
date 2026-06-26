import argparse
from pathlib import Path

import numpy as np

from visualize import read_ply


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "colmap" / "sparse_points.ply"
DEFAULT_OUTPUT = BASE_DIR / "colmap" / "sparse_points_filtered.ply"


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
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def mean_knn_distances(points, neighbors, chunk_size):
    if len(points) <= neighbors:
        raise ValueError("--neighbors must be smaller than the number of points")

    points64 = points.astype(np.float32, copy=False)
    means = np.empty(len(points64), dtype=np.float32)
    k = neighbors + 1
    for start in range(0, len(points64), chunk_size):
        stop = min(start + chunk_size, len(points64))
        chunk = points64[start:stop]
        diff = chunk[:, None, :] - points64[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32))
        nearest = np.partition(distances, k - 1, axis=1)[:, 1:k]
        means[start:stop] = nearest.mean(axis=1)
    return means


def radius_mask(points, percentile):
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    threshold = np.percentile(distances, percentile)
    return distances <= threshold, threshold


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove spatial outliers from a colored PLY point cloud."
    )
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--std-ratio", type=float, default=1.5)
    parser.add_argument(
        "--keep-percentile",
        type=float,
        default=None,
        help="Optional hard percentile for mean neighbor distance, e.g. 97 keeps the densest 97%%.",
    )
    parser.add_argument(
        "--radius-percentile",
        type=float,
        default=99.0,
        help="Remove points outside this distance percentile from the median center.",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.neighbors <= 0 or args.chunk_size <= 0:
        raise ValueError("--neighbors and --chunk-size must be positive")
    if args.std_ratio <= 0:
        raise ValueError("--std-ratio must be positive")
    if not 0 < args.radius_percentile <= 100:
        raise ValueError("--radius-percentile must be in (0, 100]")
    if args.keep_percentile is not None and not 0 < args.keep_percentile <= 100:
        raise ValueError("--keep-percentile must be in (0, 100]")

    points, colors = read_ply(args.input)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]

    neighbor_mean = mean_knn_distances(points, args.neighbors, args.chunk_size)
    threshold = float(neighbor_mean.mean() + args.std_ratio * neighbor_mean.std())
    if args.keep_percentile is not None:
        threshold = min(threshold, float(np.percentile(neighbor_mean, args.keep_percentile)))
    keep = neighbor_mean <= threshold

    radius_keep, radius_threshold = radius_mask(points, args.radius_percentile)
    keep &= radius_keep

    filtered_points = points[keep]
    filtered_colors = colors[keep]
    write_ascii_ply(args.output, filtered_points, filtered_colors)

    removed = len(points) - len(filtered_points)
    print(f"Input points: {len(points):,}")
    print(f"Output points: {len(filtered_points):,}")
    print(f"Removed points: {removed:,}")
    print(f"Mean neighbor distance threshold: {threshold:.6g}")
    print(f"Radius percentile threshold: {radius_threshold:.6g}")
    print(f"Filtered PLY: {args.output}")


if __name__ == "__main__":
    main()
