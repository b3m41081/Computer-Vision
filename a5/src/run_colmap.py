import argparse
import shutil
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


def run(command):
    print("+", " ".join(map(str, command)))
    subprocess.run([str(value) for value in command], check=True)


def option_name(colmap, command, current_name, legacy_name):
    result = subprocess.run(
        [colmap, command, "-h"],
        check=True,
        capture_output=True,
        text=True,
    )
    help_text = result.stdout + result.stderr
    return current_name if current_name in help_text else legacy_name


def supports_option(colmap, command, option):
    result = subprocess.run(
        [colmap, command, "-h"],
        check=True,
        capture_output=True,
        text=True,
    )
    return option in result.stdout + result.stderr


def image_count(image_dir):
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sum(path.suffix.lower() in extensions for path in image_dir.iterdir() if path.is_file())


def parse_args():
    parser = argparse.ArgumentParser(description="Run a standard COLMAP sparse reconstruction.")
    parser.add_argument("--images", type=Path, default=BASE_DIR / "data" / "scene" / "images_colmap_ql")
    parser.add_argument("--workspace", type=Path, default=BASE_DIR / "colmap")
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
    parser.add_argument(
        "--multiple-cameras",
        action="store_true",
        help="Estimate separate intrinsics instead of treating all images as one camera.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use CUDA for SIFT if the installed COLMAP build supports it.",
    )
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument(
        "--feature-max-image-size",
        type=int,
        default=0,
        help="Resize the longest image side for feature extraction only; 0 keeps original size.",
    )
    parser.add_argument(
        "--sift-peak-threshold",
        type=float,
        default=0.004,
        help="Lower values extract more SIFT features; COLMAP default is usually around 0.0067.",
    )
    parser.add_argument(
        "--max-orientations",
        type=int,
        default=2,
        help="Maximum orientations per SIFT feature. Higher values can create more matches.",
    )
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--sequential-overlap", type=int, default=10)
    parser.add_argument(
        "--guided-matching",
        dest="guided_matching",
        action="store_true",
        default=True,
        help="Run geometric guided matching after the initial feature matches.",
    )
    parser.add_argument(
        "--no-guided-matching",
        dest="guided_matching",
        action="store_false",
        help="Disable guided matching for a faster but usually sparser run.",
    )
    parser.add_argument(
        "--mapper-min-num-matches",
        type=int,
        default=15,
        help="Minimum verified matches for image pairs used by the mapper.",
    )
    parser.add_argument(
        "--mapper-init-min-num-inliers",
        type=int,
        default=100,
        help="Minimum inliers for the initial mapper image pair.",
    )
    parser.add_argument(
        "--keep-two-view-tracks",
        dest="keep_two_view_tracks",
        action="store_true",
        default=True,
        help="Keep two-view tracks to produce a denser sparse point cloud.",
    )
    parser.add_argument(
        "--drop-two-view-tracks",
        dest="keep_two_view_tracks",
        action="store_false",
        help="Drop two-view tracks for a cleaner but usually sparser model.",
    )
    parser.add_argument(
        "--mapper-filter-max-reproj-error",
        type=float,
        default=6.0,
        help="Larger values keep more sparse points after mapper filtering.",
    )
    parser.add_argument(
        "--mapper-filter-min-tri-angle",
        type=float,
        default=0.5,
        help="Smaller values keep points from narrower baselines.",
    )
    parser.add_argument(
        "--matcher",
        choices=("sequential", "exhaustive"),
        default="exhaustive",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--local-workspace",
        type=Path,
        default=None,
        help="Run COLMAP in local storage and publish the result to --workspace.",
    )
    return parser.parse_args()


def publish_result(source_workspace, target_workspace, overwrite):
    target_workspace.mkdir(parents=True, exist_ok=True)
    targets = {
        source_workspace / "sparse": target_workspace / "sparse",
        source_workspace / "sparse_text": target_workspace / "sparse_text",
    }
    for source, target in targets.items():
        if target.exists():
            if not overwrite:
                raise FileExistsError(f"Output already exists: {target}")
            shutil.rmtree(target)
        shutil.copytree(source, target)

    target_ply = target_workspace / "sparse_points.ply"
    if target_ply.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {target_ply}")
    shutil.copy2(source_workspace / "sparse_points.ply", target_ply)
    for database_file in ("database.db", "database.db-shm", "database.db-wal"):
        (target_workspace / database_file).unlink(missing_ok=True)
    print(f"Published reconstruction to: {target_workspace}")


def main():
    args = parse_args()
    colmap = shutil.which("colmap")
    if colmap is None:
        raise RuntimeError("COLMAP is not installed or is not available on PATH.")
    if not args.images.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.images}")
    count = image_count(args.images)
    if count < 3:
        raise RuntimeError(f"COLMAP needs a real image sequence; found only {count} images in {args.images}")
    if (
        args.max_features <= 0
        or args.num_threads <= 0
        or args.sequential_overlap <= 0
        or args.mapper_min_num_matches <= 0
        or args.mapper_init_min_num_inliers <= 0
        or args.max_orientations <= 0
    ):
        raise ValueError("Feature, thread, overlap, and mapper values must be positive")
    if (
        args.feature_max_image_size < 0
        or args.sift_peak_threshold <= 0
        or args.mapper_filter_max_reproj_error <= 0
        or args.mapper_filter_min_tri_angle <= 0
    ):
        raise ValueError("Image size and threshold values must be zero or positive")

    work_workspace = args.local_workspace
    if work_workspace is None:
        work_workspace = args.workspace

    database = work_workspace / "database.db"
    sparse = work_workspace / "sparse"
    text_model = work_workspace / "sparse_text"
    if args.overwrite:
        if database.exists():
            database.unlink()
        shutil.rmtree(sparse, ignore_errors=True)
        shutil.rmtree(text_model, ignore_errors=True)
        (work_workspace / "sparse_points.ply").unlink(missing_ok=True)
    work_workspace.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    text_model.mkdir(parents=True, exist_ok=True)
    extraction_gpu_option = option_name(
        colmap,
        "feature_extractor",
        "--FeatureExtraction.use_gpu",
        "--SiftExtraction.use_gpu",
    )
    matching_gpu_option = option_name(
        colmap,
        f"{args.matcher}_matcher",
        "--FeatureMatching.use_gpu",
        "--SiftMatching.use_gpu",
    )
    extraction_threads_option = option_name(
        colmap,
        "feature_extractor",
        "--FeatureExtraction.num_threads",
        "--SiftExtraction.num_threads",
    )
    matching_threads_option = option_name(
        colmap,
        f"{args.matcher}_matcher",
        "--FeatureMatching.num_threads",
        "--SiftMatching.num_threads",
    )

    feature_command = [
        colmap,
        "feature_extractor",
        "--database_path",
        database,
        "--image_path",
        args.images,
        "--ImageReader.camera_model",
        args.camera_model,
        "--ImageReader.single_camera",
        "0" if args.multiple_cameras else "1",
        extraction_gpu_option,
        "1" if args.use_gpu else "0",
        extraction_threads_option,
        args.num_threads,
        "--SiftExtraction.max_num_features",
        args.max_features,
        "--SiftExtraction.max_num_orientations",
        args.max_orientations,
        "--SiftExtraction.peak_threshold",
        args.sift_peak_threshold,
    ]
    if args.feature_max_image_size:
        feature_command.extend(
            ["--FeatureExtraction.max_image_size", args.feature_max_image_size]
        )
    run(feature_command)
    matcher_command = [
        colmap,
        f"{args.matcher}_matcher",
        "--database_path",
        database,
        matching_gpu_option,
        "1" if args.use_gpu else "0",
        matching_threads_option,
        args.num_threads,
    ]
    if args.matcher == "sequential":
        matcher_command.extend(["--SequentialMatching.overlap", args.sequential_overlap])
    cpu_brute_force_option = "--SiftMatching.cpu_brute_force_matcher"
    if not args.use_gpu and supports_option(
        colmap,
        f"{args.matcher}_matcher",
        cpu_brute_force_option,
    ):
        matcher_command.extend([cpu_brute_force_option, "1"])
    guided_matching_option = "--FeatureMatching.guided_matching"
    if args.guided_matching and supports_option(
        colmap,
        f"{args.matcher}_matcher",
        guided_matching_option,
    ):
        matcher_command.extend([guided_matching_option, "1"])
    run(matcher_command)
    mapper_command = [
        colmap,
        "mapper",
        "--database_path",
        database,
        "--image_path",
        args.images,
        "--output_path",
        sparse,
        "--Mapper.num_threads",
        args.num_threads,
        "--Mapper.min_num_matches",
        args.mapper_min_num_matches,
        "--Mapper.init_min_num_inliers",
        args.mapper_init_min_num_inliers,
        "--Mapper.tri_ignore_two_view_tracks",
        "0" if args.keep_two_view_tracks else "1",
    ]
    mapper_options = {
        "--Mapper.filter_max_reproj_error": args.mapper_filter_max_reproj_error,
        "--Mapper.filter_min_tri_angle": args.mapper_filter_min_tri_angle,
    }
    for option, value in mapper_options.items():
        if supports_option(colmap, "mapper", option):
            mapper_command.extend([option, value])
    run(mapper_command)

    models = sorted(path for path in sparse.iterdir() if path.is_dir())
    if not models:
        raise RuntimeError("COLMAP did not create a sparse model.")
    run(
        [
            colmap,
            "model_converter",
            "--input_path",
            models[0],
            "--output_path",
            text_model,
            "--output_type",
            "TXT",
        ]
    )
    run(
        [
            colmap,
            "model_converter",
            "--input_path",
            models[0],
            "--output_path",
            work_workspace / "sparse_points.ply",
            "--output_type",
            "PLY",
        ]
    )
    if work_workspace.resolve() != args.workspace.resolve():
        publish_result(work_workspace, args.workspace, args.overwrite)
    print(f"Sparse reconstruction: {models[0]}")


if __name__ == "__main__":
    main()
