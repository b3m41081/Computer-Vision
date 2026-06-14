import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


def running_in_docker():
    return Path("/.dockerenv").exists() or os.environ.get("A5_DOCKER") == "1"


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
    parser.add_argument("--images", type=Path, default=BASE_DIR / "data" / "scene" / "images")
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
        help="Use CUDA for SIFT. Leave disabled for Docker Desktop on macOS.",
    )
    parser.add_argument("--max-features", type=int, default=4096)
    parser.add_argument("--num-threads", type=int, default=2 if running_in_docker() else 4)
    parser.add_argument("--sequential-overlap", type=int, default=5)
    parser.add_argument(
        "--matcher",
        choices=("sequential", "exhaustive"),
        default="sequential",
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
    if args.max_features <= 0 or args.num_threads <= 0 or args.sequential_overlap <= 0:
        raise ValueError("Feature, thread, and overlap values must be positive")

    temporary_workspace = None
    work_workspace = args.local_workspace
    if work_workspace is None and running_in_docker():
        temporary_workspace = tempfile.TemporaryDirectory(prefix="a5-colmap-")
        work_workspace = Path(temporary_workspace.name)
        print(
            "Docker detected: using container-local storage for the COLMAP database "
            "to avoid SQLite errors on macOS bind mounts."
        )
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

    run(
        [
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
        ]
    )
    matcher_command = [
        colmap,
        f"{args.matcher}_matcher",
        "--database_path",
        database,
        matching_gpu_option,
        "1" if args.use_gpu else "0",
        matching_threads_option,
        1 if not args.use_gpu else args.num_threads,
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
    run(matcher_command)
    run(
        [
            colmap,
            "mapper",
            "--database_path",
            database,
            "--image_path",
            args.images,
            "--output_path",
            sparse,
        ]
    )

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
    if temporary_workspace is not None:
        temporary_workspace.cleanup()


if __name__ == "__main__":
    main()
