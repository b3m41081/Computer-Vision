import argparse
import contextlib
import gc
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = BASE_DIR / "data" / "scene" / "images"
DEFAULT_OUTPUT_DIR = BASE_DIR / "da3"
DEFAULT_DA3_REPO = BASE_DIR / "vendor" / "depth-anything-3"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Depth Anything 3 and export a fused colored point cloud."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--da3-repo",
        type=Path,
        default=Path(os.environ.get("DA3_REPO", DEFAULT_DA3_REPO)),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default=os.environ.get("DA3_DEVICE", "auto"),
    )
    parser.add_argument("--model", default=os.environ.get("DA3_MODEL", "depth-anything/DA3-SMALL"))
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=500000)
    parser.add_argument("--confidence-percentile", type=float, default=20.0)
    parser.add_argument(
        "--resolution",
        type=int,
        default=0,
        help="Processing resolution. Auto: 392 on MPS, 504 on CUDA, 336 on CPU.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def add_da3_to_path(repo):
    source = repo / "src"
    if source.is_dir():
        sys.path.insert(0, str(source.resolve()))


def select_images(image_dir, max_images):
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No input images found in: {image_dir}")
    if len(images) <= max_images:
        return images
    if max_images == 1:
        return [images[len(images) // 2]]
    indices = np.linspace(0, len(images) - 1, max_images, dtype=np.int64)
    return [images[index] for index in indices]


def choose_device(torch, requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available to PyTorch.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available to PyTorch.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def processing_resolution(requested, device):
    resolution = requested or {"cuda": 504, "mps": 392, "cpu": 336}[device.type]
    if resolution < 224 or resolution > 1008 or resolution % 14:
        raise ValueError("Resolution must be 0 (auto) or a multiple of 14 from 224 to 1008.")
    return resolution


def release_device_memory(torch, device):
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def force_fp32_forward_on_mps(model, torch):
    original_autocast = torch.autocast

    def compatible_autocast(device_type, *args, **kwargs):
        if device_type == "mps" and kwargs.get("enabled") is False:
            return contextlib.nullcontext()
        return original_autocast(device_type, *args, **kwargs)

    torch.autocast = compatible_autocast

    def forward_fp32(
        self,
        image,
        extrinsics=None,
        intrinsics=None,
        export_feat_layers=None,
        infer_gs=False,
        use_ray_pose=False,
        ref_view_strategy="saddle_balanced",
    ):
        with torch.inference_mode(), contextlib.nullcontext():
            return self.model(
                image,
                extrinsics,
                intrinsics,
                export_feat_layers,
                infer_gs,
                use_ray_pose,
                ref_view_strategy,
            )

    model.forward = types.MethodType(forward_fp32, model)


def as_homogeneous(extrinsic):
    if extrinsic.shape == (4, 4):
        return extrinsic
    result = np.eye(4, dtype=extrinsic.dtype)
    result[:3, :4] = extrinsic
    return result


def prediction_to_points(prediction, confidence_percentile):
    if prediction.intrinsics is None or prediction.extrinsics is None:
        raise RuntimeError("DA3 did not return camera intrinsics and extrinsics.")
    confidence = prediction.conf
    threshold = (
        float(np.percentile(confidence[np.isfinite(confidence)], confidence_percentile))
        if confidence is not None
        else None
    )
    depth = prediction.depth
    images = prediction.processed_images
    _, height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)
    all_points = []
    all_colors = []

    for index in range(len(depth)):
        valid = np.isfinite(depth[index]) & (depth[index] > 0)
        if confidence is not None:
            valid &= np.isfinite(confidence[index]) & (confidence[index] >= threshold)
        selected = np.flatnonzero(valid.reshape(-1))
        if not len(selected):
            continue
        rays = np.linalg.inv(prediction.intrinsics[index]) @ pixels[selected].T
        camera_points = rays * depth[index].reshape(-1)[selected][None, :]
        camera_points_h = np.vstack([camera_points, np.ones((1, len(selected)))])
        camera_to_world = np.linalg.inv(as_homogeneous(prediction.extrinsics[index]))
        world_points = (camera_to_world @ camera_points_h)[:3].T.astype(np.float32)
        colors = images[index].reshape(-1, 3)[selected].astype(np.uint8)
        all_points.append(world_points)
        all_colors.append(colors)

    if not all_points:
        raise RuntimeError("DA3 produced no valid points at the selected confidence percentile.")
    return np.concatenate(all_points), np.concatenate(all_colors), threshold


def filter_and_sample(points, colors, max_points, seed):
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    limit = np.percentile(distances, 99.5)
    inliers = distances <= limit
    points, colors = points[inliers], colors[inliers]
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(points), max_points, replace=False)
        points, colors = points[indices], colors[indices]
    return points.astype(np.float32), colors


def write_binary_ply(path, points, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as file:
        file.write(header)
        vertices.tofile(file)


def source_video_for(image_dir):
    manifest_path = image_dir.parent / "video_frames.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    source = manifest.get("source_video")
    return Path(source).name if source else None


def main():
    args = parse_args()
    if args.max_images <= 0 or args.max_points <= 0:
        raise ValueError("Image and point limits must be positive.")
    if not 0 <= args.confidence_percentile < 100:
        raise ValueError("Confidence percentile must be in [0, 100).")

    add_da3_to_path(args.da3_repo)
    # The official API eagerly imports its optional COLMAP exporter. pycolmap
    # has no ARM64 Linux wheel and is not used by this runner's PLY export.
    if "pycolmap" not in sys.modules:
        try:
            import pycolmap  # noqa: F401
        except ImportError:
            sys.modules["pycolmap"] = types.ModuleType("pycolmap")
    try:
        import torch
        from depth_anything_3.api import DepthAnything3
    except ImportError as error:
        raise RuntimeError(
            "Depth Anything 3 is not installed. Follow the A5 README DA3 setup."
        ) from error

    selected = select_images(args.images, args.max_images)
    device = choose_device(torch, args.device)
    resolution = processing_resolution(args.resolution, device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"DA3 model: {args.model}", flush=True)
    print(f"DA3 device: {device}", flush=True)
    print(f"DA3 resolution: {resolution}", flush=True)
    print(f"Selected images: {len(selected)}", flush=True)
    for path in selected:
        print(f"  - {path.name}", flush=True)

    started = time.time()
    print("Loading DA3 checkpoint...", flush=True)
    model = DepthAnything3.from_pretrained(args.model).eval().to(device)
    if device.type == "mps":
        force_fp32_forward_on_mps(model, torch)
    print("Running multi-view depth and camera prediction...", flush=True)
    prediction = model.inference(
        [str(path) for path in selected],
        process_res=resolution,
        process_res_method="upper_bound_resize",
        ref_view_strategy="middle",
    )
    del model
    release_device_memory(torch, device)

    points, colors, confidence_threshold = prediction_to_points(
        prediction, args.confidence_percentile
    )
    points, colors = filter_and_sample(points, colors, args.max_points, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_binary_ply(args.output_dir / "points.ply", points, colors)
    np.savez_compressed(
        args.output_dir / "cameras.npz",
        extrinsic=prediction.extrinsics,
        intrinsic=prediction.intrinsics,
        image_names=np.asarray([path.name for path in selected]),
    )
    metadata = {
        "model": args.model,
        "device": str(device),
        "input_image_count": len(selected),
        "input_images": [path.name for path in selected],
        "source_video": source_video_for(args.images),
        "confidence_percentile": args.confidence_percentile,
        "confidence_threshold": (
            round(float(confidence_threshold), 6)
            if confidence_threshold is not None
            else None
        ),
        "point_count": len(points),
        "resolution": resolution,
        "is_metric": bool(prediction.is_metric),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"DA3 point cloud: {args.output_dir / 'points.ply'} ({len(points):,} points)")
    print(f"DA3 metadata: {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
