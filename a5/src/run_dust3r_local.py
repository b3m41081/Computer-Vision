import argparse
import json
import sys
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = BASE_DIR / "data" / "scene" / "images_colmap_ql"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "dust3r_local"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def import_dust3r(dust3r_repo):
    if dust3r_repo is not None:
        sys.path.insert(0, str(dust3r_repo.resolve()))
    try:
        import torch
        from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.image import load_images
    except ImportError as error:
        raise RuntimeError(
            "DUSt3R is not importable. Clone https://github.com/naver/dust3r, "
            "install its requirements in the active environment, then pass "
            "--dust3r-repo /path/to/dust3r if needed."
        ) from error
    return torch, AsymmetricCroCo3DStereo, load_images, make_pairs, inference, global_aligner, GlobalAlignerMode


def choose_device(torch, requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def image_paths(image_dir, limit):
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit is not None:
        paths = paths[:limit]
    if len(paths) < 2:
        raise RuntimeError(f"DUSt3R needs at least two images; found {len(paths)} in {image_dir}")
    return paths


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def colors_from_image(image, mask):
    colors = as_numpy(image)
    if colors.ndim == 3 and colors.shape[0] == 3:
        colors = np.moveaxis(colors, 0, -1)
    if colors.max(initial=0) <= 1.0:
        colors = colors * 255.0
    return np.clip(colors.reshape(-1, 3)[mask], 0, 255).astype(np.uint8)


def point_cloud_from_scene(torch, scene, min_confidence):
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_confidence)))
    points_per_view = scene.get_pts3d()
    masks_per_view = scene.get_masks()
    images = scene.imgs

    all_points = []
    all_colors = []
    for points, mask, image in zip(points_per_view, masks_per_view, images):
        points = as_numpy(points).reshape(-1, 3)
        mask = as_numpy(mask).astype(bool).reshape(-1)
        all_points.append(points[mask].astype(np.float32))
        all_colors.append(colors_from_image(image, mask))

    if not all_points:
        raise RuntimeError("DUSt3R did not return any point cloud data")
    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    colors = colors[valid]
    if len(points) == 0:
        raise RuntimeError("All exported DUSt3R points were invalid")
    return points, colors


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
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Run DUSt3R locally on the prepared A5 images.")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dust3r-repo", type=Path, default=None)
    parser.add_argument(
        "--model",
        default="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        help="Hugging Face model name or local checkpoint understood by DUSt3R.",
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--image-size", type=int, choices=(224, 512), default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--schedule", choices=("linear", "cosine"), default="cosine")
    parser.add_argument("--scene-graph", default="complete")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=3.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.images.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.images}")
    if args.batch_size <= 0 or args.niter <= 0 or args.lr <= 0:
        raise ValueError("--batch-size, --niter, and --lr must be positive")

    (
        torch,
        AsymmetricCroCo3DStereo,
        load_images,
        make_pairs,
        inference,
        global_aligner,
        GlobalAlignerMode,
    ) = import_dust3r(args.dust3r_repo)
    device = choose_device(torch, args.device)
    paths = image_paths(args.images, args.max_images)

    print(f"Loading DUSt3R model on {device}: {args.model}")
    model = AsymmetricCroCo3DStereo.from_pretrained(args.model).to(device)
    try:
        square_ok = model.square_ok
    except AttributeError:
        square_ok = False
    images = load_images(
        [str(path) for path in paths],
        size=args.image_size,
        patch_size=model.patch_size,
        square_ok=square_ok,
    )
    pairs = make_pairs(images, scene_graph=args.scene_graph, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=args.batch_size)

    mode = (
        GlobalAlignerMode.PointCloudOptimizer
        if len(paths) > 2
        else GlobalAlignerMode.PairViewer
    )
    scene = global_aligner(output, device=device, mode=mode)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init="mst",
            niter=args.niter,
            schedule=args.schedule,
            lr=args.lr,
        )
        print(f"Global alignment loss: {float(loss):.6f}")

    points, colors = point_cloud_from_scene(torch, scene, args.min_confidence)
    ply_path = args.output_dir / "dust3r_points.ply"
    write_ascii_ply(ply_path, points, colors)

    metadata = {
        "model": args.model,
        "device": device,
        "image_size": args.image_size,
        "scene_graph": args.scene_graph,
        "image_count": len(paths),
        "point_count": int(len(points)),
        "images": [str(path) for path in paths],
        "output": str(ply_path),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"DUSt3R point cloud: {ply_path}")


if __name__ == "__main__":
    main()
