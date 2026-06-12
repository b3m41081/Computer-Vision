import argparse
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = BASE_DIR / "output" / "final_reconstruction.ply"
DEFAULT_SCREENSHOT = BASE_DIR / "output" / "visualization.png"


def read_ascii_ply(path):
    with path.open("r", encoding="ascii") as file:
        if file.readline().strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        vertex_count = None
        properties = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError("PLY header ended unexpectedly")
            fields = line.strip().split()
            if fields[:2] == ["format", "ascii"]:
                continue
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
            elif fields[:1] == ["property"] and vertex_count is not None:
                properties.append(fields[-1])
            elif fields[:1] == ["end_header"]:
                break

        if vertex_count is None:
            raise ValueError("PLY file has no vertex element")
        required = ["x", "y", "z", "red", "green", "blue"]
        if any(name not in properties for name in required):
            raise ValueError(f"PLY must contain these vertex properties: {required}")

        data = np.loadtxt(file, dtype=np.float32, max_rows=vertex_count)

    if data.ndim == 1:
        data = data[None, :]
    indices = {name: properties.index(name) for name in required}
    points = data[:, [indices["x"], indices["y"], indices["z"]]]
    colors = data[:, [indices["red"], indices["green"], indices["blue"]]].astype(np.uint8)
    return points, colors


def rotation_matrix(yaw_deg, pitch_deg):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    rotation_y = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]],
        dtype=np.float32,
    )
    rotation_x = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]],
        dtype=np.float32,
    )
    return rotation_x @ rotation_y


def normalize_cloud(points):
    center = np.median(points, axis=0)
    centered = points - center
    radius = float(np.percentile(np.linalg.norm(centered, axis=1), 95))
    if radius <= 0:
        raise ValueError("Point cloud has zero spatial extent")
    return centered / radius


def render(points, colors, width, height, yaw, pitch, zoom):
    canvas = np.full((height, width, 3), (18, 20, 25), dtype=np.uint8)
    transformed = points @ rotation_matrix(yaw, pitch).T
    camera_distance = 3.2
    denominator = transformed[:, 2] + camera_distance
    visible = denominator > 0.1
    transformed = transformed[visible]
    denominator = denominator[visible]
    rgb = colors[visible]

    focal = min(width, height) * zoom
    px = np.rint(width / 2 + focal * transformed[:, 0] / denominator).astype(np.int32)
    py = np.rint(height / 2 - focal * transformed[:, 1] / denominator).astype(np.int32)
    inside = (px >= 2) & (px < width - 2) & (py >= 58) & (py < height - 2)
    px, py = px[inside], py[inside]
    rgb = rgb[inside]
    z = transformed[inside, 2]

    order = np.argsort(z)[::-1]
    px, py, rgb = px[order], py[order], rgb[order]
    bgr = rgb[:, ::-1]
    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        canvas[py + dy, px + dx] = bgr

    cv2.rectangle(canvas, (0, 0), (width, 50), (8, 10, 14), -1)
    cv2.putText(
        canvas,
        "A5 - colored 3D reconstruction",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (235, 238, 245),
        2,
        cv2.LINE_AA,
    )
    return canvas


def run_interactive(points, colors, width, height, yaw, pitch, zoom):
    while True:
        frame = render(points, colors, width, height, yaw, pitch, zoom)
        cv2.imshow("A5 point cloud", frame)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("a"):
            yaw -= 5
        elif key == ord("d"):
            yaw += 5
        elif key == ord("w"):
            pitch -= 5
        elif key == ord("s"):
            pitch += 5
        elif key in (ord("+"), ord("=")):
            zoom *= 1.1
        elif key == ord("-"):
            zoom /= 1.1
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Render an ASCII colored PLY without extra 3D dependencies.")
    parser.add_argument("model", type=Path, nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--screenshot", type=Path, default=DEFAULT_SCREENSHOT)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--yaw", type=float, default=-12.0)
    parser.add_argument("--pitch", type=float, default=-5.0)
    parser.add_argument("--zoom", type=float, default=3.0)
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    points, colors = read_ascii_ply(args.model)
    points = normalize_cloud(points)
    frame = render(points, colors, args.width, args.height, args.yaw, args.pitch, args.zoom)
    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.screenshot), frame):
        raise RuntimeError(f"Could not write screenshot: {args.screenshot}")
    print(f"Rendered {len(points):,} points to {args.screenshot}")

    if args.interactive:
        run_interactive(points, colors, args.width, args.height, args.yaw, args.pitch, args.zoom)


if __name__ == "__main__":
    main()
