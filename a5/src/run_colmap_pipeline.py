import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BASE_DIR.parent
DEFAULT_IMAGE_DIR = BASE_DIR / "data" / "scene" / "images_colmap_ql"
DEFAULT_WORKSPACE = BASE_DIR / "colmap"
DEFAULT_SCREENSHOT = BASE_DIR / "img" / "colmap_sparse.png"


def run(command):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=REPO_DIR, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the A5 sparse COLMAP reconstruction from prepared images."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--screenshot", type=Path, default=DEFAULT_SCREENSHOT)
    parser.add_argument("--skip-colmap", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--feature-max-image-size", type=int, default=0)
    parser.add_argument("--sift-peak-threshold", type=float, default=0.004)
    parser.add_argument("--max-orientations", type=int, default=2)
    parser.add_argument("--sequential-overlap", type=int, default=10)
    parser.add_argument(
        "--guided-matching",
        dest="guided_matching",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-guided-matching",
        dest="guided_matching",
        action="store_false",
    )
    parser.add_argument("--mapper-min-num-matches", type=int, default=15)
    parser.add_argument("--mapper-init-min-num-inliers", type=int, default=100)
    parser.add_argument(
        "--keep-two-view-tracks",
        dest="keep_two_view_tracks",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--drop-two-view-tracks",
        dest="keep_two_view_tracks",
        action="store_false",
    )
    parser.add_argument("--mapper-filter-max-reproj-error", type=float, default=6.0)
    parser.add_argument("--mapper-filter-min-tri-angle", type=float, default=0.5)
    parser.add_argument(
        "--matcher",
        choices=("sequential", "exhaustive"),
        default="exhaustive",
    )
    parser.add_argument("--multiple-cameras", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    if not args.skip_colmap:
        command = [
            sys.executable,
            BASE_DIR / "src" / "run_colmap.py",
            "--images",
            args.images,
            "--workspace",
            args.workspace,
            "--overwrite",
            "--max-features",
            args.max_features,
            "--feature-max-image-size",
            args.feature_max_image_size,
            "--sift-peak-threshold",
            args.sift_peak_threshold,
            "--max-orientations",
            args.max_orientations,
            "--sequential-overlap",
            args.sequential_overlap,
            "--num-threads",
            args.threads,
            "--matcher",
            args.matcher,
            "--mapper-min-num-matches",
            args.mapper_min_num_matches,
            "--mapper-init-min-num-inliers",
            args.mapper_init_min_num_inliers,
            "--mapper-filter-max-reproj-error",
            args.mapper_filter_max_reproj_error,
            "--mapper-filter-min-tri-angle",
            args.mapper_filter_min_tri_angle,
        ]
        if args.guided_matching:
            command.append("--guided-matching")
        if args.keep_two_view_tracks:
            command.append("--keep-two-view-tracks")
        if args.multiple_cameras:
            command.append("--multiple-cameras")
        if args.use_gpu:
            command.append("--use-gpu")
        run(command)

    if not args.skip_render:
        command = [
            sys.executable,
            BASE_DIR / "src" / "visualize.py",
            args.workspace / "sparse_points.ply",
            "--cameras",
            args.workspace / "sparse_text" / "images.txt",
            "--screenshot",
            args.screenshot,
        ]
        if args.interactive:
            command.append("--interactive")
        run(command)

    print("A5 COLMAP image pipeline finished.")


if __name__ == "__main__":
    main()
