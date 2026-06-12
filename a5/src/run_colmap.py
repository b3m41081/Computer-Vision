import argparse
import shutil
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


def run(command):
    print("+", " ".join(map(str, command)))
    subprocess.run([str(value) for value in command], check=True)


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
    return parser.parse_args()


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

    database = args.workspace / "database.db"
    sparse = args.workspace / "sparse"
    text_model = args.workspace / "sparse_text"
    args.workspace.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    text_model.mkdir(parents=True, exist_ok=True)

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
        ]
    )
    run([colmap, "exhaustive_matcher", "--database_path", database])
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
            args.workspace / "sparse_points.ply",
            "--output_type",
            "PLY",
        ]
    )
    print(f"Sparse reconstruction: {models[0]}")


if __name__ == "__main__":
    main()
