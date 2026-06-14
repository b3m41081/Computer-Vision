import argparse
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = BASE_DIR / "colmap" / "sparse_points.ply"
DEFAULT_SCREENSHOT = BASE_DIR / "img" / "colmap_sparse.png"


PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def read_ply(path):
    with path.open("rb") as file:
        if file.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")

        ply_format = None
        vertex_count = None
        properties = []
        current_element = None
        while True:
            raw_line = file.readline()
            line = raw_line.decode("ascii").strip()
            if not line:
                raise ValueError("PLY header ended unexpectedly")
            fields = line.split()
            if fields[:1] == ["format"]:
                ply_format = fields[1]
            elif fields[:1] == ["element"]:
                current_element = fields[1]
                if current_element == "vertex":
                    vertex_count = int(fields[2])
            elif fields[:1] == ["property"] and current_element == "vertex":
                if fields[1] == "list":
                    raise ValueError("List properties in the vertex element are unsupported")
                properties.append((fields[1], fields[2]))
            elif fields[:1] == ["end_header"]:
                break

        if vertex_count is None:
            raise ValueError("PLY file has no vertex element")
        required = ["x", "y", "z", "red", "green", "blue"]
        property_names = [name for _, name in properties]
        if any(name not in property_names for name in required):
            raise ValueError(f"PLY must contain these vertex properties: {required}")

        if ply_format == "ascii":
            data = np.loadtxt(file, dtype=np.float64, max_rows=vertex_count)
            if data.ndim == 1:
                data = data[None, :]
            columns = {name: data[:, index] for index, (_, name) in enumerate(properties)}
        elif ply_format in ("binary_little_endian", "binary_big_endian"):
            endian = "<" if ply_format == "binary_little_endian" else ">"
            try:
                dtype = np.dtype(
                    [(name, endian + PLY_TYPES[property_type]) for property_type, name in properties]
                )
            except KeyError as error:
                raise ValueError(f"Unsupported PLY property type: {error.args[0]}") from error
            data = np.fromfile(file, dtype=dtype, count=vertex_count)
            columns = {name: data[name] for _, name in properties}
        else:
            raise ValueError(f"Unsupported PLY format: {ply_format}")

    if len(data) != vertex_count:
        raise ValueError(f"Expected {vertex_count} vertices, found {len(data)}")
    points = np.column_stack([columns[name] for name in ("x", "y", "z")]).astype(np.float32)
    colors = np.column_stack([columns[name] for name in ("red", "green", "blue")]).astype(
        np.uint8
    )
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
    distances = np.linalg.norm(centered, axis=1)
    inliers = centered[distances <= np.percentile(distances, 90)]
    _, _, principal_axes = np.linalg.svd(inliers, full_matrices=False)
    centered = centered @ principal_axes.T
    radius = float(np.percentile(np.linalg.norm(centered, axis=1), 95))
    if radius <= 0:
        raise ValueError("Point cloud has zero spatial extent")
    return centered / radius


def render(points, colors, width, height, yaw, pitch, zoom, title, point_radius=1):
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
    offsets = [
        (dx, dy)
        for dy in range(-point_radius, point_radius + 1)
        for dx in range(-point_radius, point_radius + 1)
        if dx * dx + dy * dy <= point_radius * point_radius
    ]
    for dx, dy in offsets:
        canvas[py + dy, px + dx] = bgr

    cv2.rectangle(canvas, (0, 0), (width, 50), (8, 10, 14), -1)
    cv2.putText(
        canvas,
        title,
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (235, 238, 245),
        2,
        cv2.LINE_AA,
    )
    return canvas


def run_interactive(points, colors, width, height, yaw, pitch, zoom, title, point_radius):
    while True:
        frame = render(
            points, colors, width, height, yaw, pitch, zoom, title, point_radius
        )
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
    parser = argparse.ArgumentParser(description="Render a colored PLY without extra 3D dependencies.")
    parser.add_argument("model", type=Path, nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--screenshot", type=Path, default=DEFAULT_SCREENSHOT)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--zoom", type=float, default=3.5)
    parser.add_argument("--title", default="A5 - COLMAP sparse reconstruction")
    parser.add_argument("--point-radius", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    points, colors = read_ply(args.model)
    points = normalize_cloud(points)
    frame = render(
        points,
        colors,
        args.width,
        args.height,
        args.yaw,
        args.pitch,
        args.zoom,
        args.title,
        args.point_radius,
    )
    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.screenshot), frame):
        raise RuntimeError(f"Could not write screenshot: {args.screenshot}")
    print(f"Rendered {len(points):,} points to {args.screenshot}")

    if args.interactive:
        run_interactive(
            points,
            colors,
            args.width,
            args.height,
            args.yaw,
            args.pitch,
            args.zoom,
            args.title,
            args.point_radius,
        )


if __name__ == "__main__":
    main()
