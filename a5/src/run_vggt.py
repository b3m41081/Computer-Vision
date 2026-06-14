import argparse
import contextlib
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = BASE_DIR / "data" / "scene" / "images"
DEFAULT_OUTPUT_DIR = BASE_DIR / "vggt"
DEFAULT_VGGT_REPO = BASE_DIR / "vendor" / "vggt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VGGT on an image sequence and export a colored point cloud."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--vggt-repo",
        type=Path,
        default=Path(os.environ.get("VGGT_REPO", DEFAULT_VGGT_REPO)),
        help="Official facebookresearch/vggt checkout. Optional if vggt is installed.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default=os.environ.get("VGGT_DEVICE", "auto"),
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--confidence-threshold", type=float, default=1.0)
    parser.add_argument(
        "--resolution",
        type=int,
        default=0,
        help="Square inference size divisible by 14. Auto: 392 on MPS, 294 on CPU, 518 on CUDA.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="facebook/VGGT-1B")
    return parser.parse_args()


def add_vggt_to_path(repo):
    if repo.is_dir():
        sys.path.insert(0, str(repo.resolve()))


def select_images(image_dir, max_images):
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    images = sorted(
        path for path in image_dir.iterdir()
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


def visible_memory_gib():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (AttributeError, OSError, ValueError):
        return None


def precision_settings(torch, device, requested):
    if requested == "fp32" or device.type == "cpu":
        return torch.float32, contextlib.nullcontext()
    if device.type == "mps":
        if requested in ("fp16", "bf16"):
            raise ValueError(
                "The current VGGT depth head requires fp32 on MPS; use auto or fp32."
            )
        return torch.float32, contextlib.nullcontext()
    if requested == "bf16":
        dtype = torch.bfloat16
    elif requested == "fp16":
        dtype = torch.float16
    elif device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    return dtype, torch.autocast(device_type=device.type, dtype=dtype)


def inference_resolution(requested, device):
    automatic = {"cuda": 518, "mps": 392, "cpu": 294}
    resolution = requested or automatic[device.type]
    if resolution < 196 or resolution > 518 or resolution % 14 != 0:
        raise ValueError("Resolution must be 0 (auto) or a multiple of 14 from 196 to 518.")
    return resolution


def release_device_memory(torch, device):
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


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
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as file:
        file.write(header)
        vertices.tofile(file)


def main():
    args = parse_args()
    if args.max_images <= 0 or args.max_points <= 0 or args.resolution < 0:
        raise ValueError("Image and point limits must be positive; resolution cannot be negative.")

    add_vggt_to_path(args.vggt_repo)
    try:
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_and_preprocess_images_square
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError as error:
        raise RuntimeError(
            "VGGT is not installed. Follow the A5 README setup or set VGGT_REPO "
            "to an official facebookresearch/vggt checkout."
        ) from error

    selected = select_images(args.images, args.max_images)
    device = choose_device(torch, args.device)
    memory_gib = visible_memory_gib()
    # Docker's nominal 8 GB allocation is reported as roughly 7.75 GiB in the VM.
    if device.type == "cpu" and memory_gib is not None and memory_gib < 7.5:
        docker_hint = ""
        if Path("/.dockerenv").is_file():
            docker_hint = (
                " Docker on macOS cannot use Apple MPS. Run VGGT natively with "
                "'.venv-vggt/bin/python a5/src/run_vggt.py --device mps --max-images 1' "
                "or increase Docker Desktop memory."
            )
        raise RuntimeError(
            f"VGGT CPU inference sees only {memory_gib:.1f} GiB RAM. "
            "It needs an 8 GB Docker allocation and is intended for native MPS or CUDA."
            f"{docker_hint}"
        )
    dtype, autocast_context = precision_settings(torch, device, args.precision)
    resolution = inference_resolution(args.resolution, device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"VGGT device: {device}", flush=True)
    print(f"VGGT precision: {dtype}", flush=True)
    print(f"VGGT resolution: {resolution} x {resolution}", flush=True)
    total_images = len(
        [
            path for path in args.images.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )
    print(f"Selected images: {len(selected)} / {total_images}", flush=True)
    for path in selected:
        print(f"  - {path.name}", flush=True)

    started = time.time()
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model = VGGT.from_pretrained(args.checkpoint).eval()
    if device.type in ("mps", "cpu"):
        # This exporter only uses cameras and depth. Removing the point and track
        # heads before the device transfer saves substantial unified memory.
        model.point_head = None
        model.track_head = None
        release_device_memory(torch, device)
    model = model.to(device)
    images, original_coords = load_and_preprocess_images_square(
        [str(path) for path in selected], resolution
    )
    inference_images = images.to(device)
    del images
    release_device_memory(torch, device)

    print("Running VGGT camera and depth prediction...", flush=True)
    with torch.inference_mode(), autocast_context:
        batch = inference_images[None]
        aggregated_tokens, patch_start_index = model.aggregator(batch)
        print("VGGT transformer finished.", flush=True)
        pose_encoding = model.camera_head(aggregated_tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_encoding, batch.shape[-2:])
        if device.type in ("mps", "cpu"):
            model.aggregator = None
            model.camera_head = None
            del pose_encoding
            release_device_memory(torch, device)
        print("Running VGGT depth head...", flush=True)
        depth_map, depth_confidence = model.depth_head(
            aggregated_tokens, batch, patch_start_index, frames_chunk_size=1
        )
        print("VGGT depth prediction finished.", flush=True)

    extrinsic = extrinsic.squeeze(0).float().cpu().numpy()
    intrinsic = intrinsic.squeeze(0).float().cpu().numpy()
    depth_map = depth_map.squeeze(0).float().cpu().numpy()
    confidence = depth_confidence.squeeze(0).float().cpu().numpy()
    points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    colors = (
        inference_images.detach().float().clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
        * 255
    ).astype(np.uint8)

    confidence = confidence.reshape(points.shape[:-1])
    depth_values = depth_map.reshape(points.shape[:-1])
    base_valid = (
        np.isfinite(points).all(axis=-1)
        & np.isfinite(confidence)
        & np.isfinite(depth_values)
        & (depth_values > 0)
    )
    if not np.any(base_valid):
        raise RuntimeError("VGGT produced no finite points with positive depth.")
    effective_threshold = args.confidence_threshold
    valid = base_valid & (confidence >= effective_threshold)
    if not np.any(valid):
        effective_threshold = float(np.percentile(confidence[base_valid], 90))
        valid = base_valid & (confidence >= effective_threshold)
        print(
            f"Confidence threshold {args.confidence_threshold:g} removed all points; "
            f"using the top 10% at {effective_threshold:.3f} instead."
        )
    points = points[valid].astype(np.float32)
    colors = colors[valid]
    if len(points) > args.max_points:
        indices = np.random.choice(len(points), args.max_points, replace=False)
        points, colors = points[indices], colors[indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    point_cloud = args.output_dir / "points.ply"
    write_binary_ply(point_cloud, points, colors)
    np.savez_compressed(
        args.output_dir / "cameras.npz",
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        image_names=np.asarray([path.name for path in selected]),
        original_coords=original_coords.cpu().numpy(),
    )
    metadata = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "precision": str(dtype).replace("torch.", ""),
        "input_image_count": len(selected),
        "input_images": [path.name for path in selected],
        "confidence_threshold": args.confidence_threshold,
        "effective_confidence_threshold": round(effective_threshold, 6),
        "point_count": len(points),
        "resolution": resolution,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"VGGT point cloud: {point_cloud} ({len(points):,} points)")
    print(f"VGGT metadata: {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
