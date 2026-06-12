import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the official VGGT COLMAP export script.")
    parser.add_argument("--vggt-repo", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, default=BASE_DIR / "data" / "scene")
    parser.add_argument("--use-ba", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main():
    args = parse_args()
    demo = args.vggt_repo / "demo_colmap.py"
    if not demo.is_file():
        raise FileNotFoundError(f"Could not find official VGGT script: {demo}")
    image_dir = args.scene_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Expected input images in: {image_dir}")

    command = [args.python, str(demo), f"--scene_dir={args.scene_dir}"]
    if args.use_ba:
        command.append("--use_ba")
    print("+", " ".join(command))
    subprocess.run(command, check=True, cwd=args.vggt_repo)


if __name__ == "__main__":
    main()
