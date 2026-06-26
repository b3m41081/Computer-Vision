import argparse
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = BASE_DIR / "colmap" / "sparse_points.ply"
DEFAULT_CAMERAS = BASE_DIR / "colmap" / "sparse_text" / "images.txt"
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


def qvec_to_rotation(qvec):
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ],
        dtype=np.float32,
    )


def read_colmap_camera_positions(path):
    if not path.is_file():
        return None, []

    records = []
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        if len(fields) < 10:
            continue
        qvec = np.asarray([float(value) for value in fields[1:5]], dtype=np.float32)
        tvec = np.asarray([float(value) for value in fields[5:8]], dtype=np.float32)
        rotation = qvec_to_rotation(qvec)
        center = -(rotation.T @ tvec)
        records.append({"name": fields[9], "center": center})

    if not records:
        return None, []
    centers = np.stack([record["center"] for record in records]).astype(np.float32)
    return centers, records


def rotation_matrix(yaw_deg, pitch_deg, roll_deg=0.0):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)
    rotation_y = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]],
        dtype=np.float32,
    )
    rotation_x = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]],
        dtype=np.float32,
    )
    rotation_z = np.array(
        [[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    return rotation_z @ rotation_x @ rotation_y


def normalize_scene(points, camera_positions=None):
    center = np.median(points, axis=0)
    centered = points - center
    distances = np.linalg.norm(centered, axis=1)
    inliers = centered[distances <= np.percentile(distances, 90)]
    _, _, principal_axes = np.linalg.svd(inliers, full_matrices=False)
    centered = centered @ principal_axes.T
    radius = float(np.percentile(np.linalg.norm(centered, axis=1), 95))
    if radius <= 0:
        raise ValueError("Point cloud has zero spatial extent")
    normalized_points = centered / radius
    normalized_cameras = None
    if camera_positions is not None:
        normalized_cameras = ((camera_positions - center) @ principal_axes.T) / radius
    return normalized_points, normalized_cameras


def project_points(points, width, height, yaw, pitch, roll, zoom):
    transformed = points @ rotation_matrix(yaw, pitch, roll).T
    camera_distance = 3.2
    denominator = transformed[:, 2] + camera_distance
    visible = denominator > 0.1
    focal = min(width, height) * zoom
    px = np.zeros(len(points), dtype=np.int32)
    py = np.zeros(len(points), dtype=np.int32)
    px[visible] = np.rint(
        width / 2 + focal * transformed[visible, 0] / denominator[visible]
    ).astype(np.int32)
    py[visible] = np.rint(
        height / 2 - focal * transformed[visible, 1] / denominator[visible]
    ).astype(np.int32)
    return px, py, transformed[:, 2], visible


def render(
    points,
    colors,
    camera_positions,
    width,
    height,
    yaw,
    pitch,
    roll,
    zoom,
    title,
    point_radius=1,
    show_cameras=True,
):
    canvas = np.full((height, width, 3), (18, 20, 25), dtype=np.uint8)
    px, py, z_values, visible = project_points(points, width, height, yaw, pitch, roll, zoom)
    rgb = colors[visible]
    px, py = px[visible], py[visible]
    inside = (px >= 2) & (px < width - 2) & (py >= 58) & (py < height - 2)
    px, py = px[inside], py[inside]
    rgb = rgb[inside]
    z = z_values[visible][inside]

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

    camera_count = 0
    if show_cameras and camera_positions is not None and len(camera_positions):
        cx, cy, cz, camera_visible = project_points(
            camera_positions, width, height, yaw, pitch, roll, zoom
        )
        camera_inside = (
            camera_visible
            & (cx >= 8)
            & (cx < width - 8)
            & (cy >= 62)
            & (cy < height - 8)
        )
        camera_indices = np.flatnonzero(camera_inside)
        camera_count = len(camera_indices)
        if camera_count > 1:
            path_points = np.column_stack([cx[camera_indices], cy[camera_indices]])
            for start, end in zip(path_points[:-1], path_points[1:]):
                cv2.line(
                    canvas,
                    tuple(start.astype(int)),
                    tuple(end.astype(int)),
                    (70, 150, 255),
                    1,
                    cv2.LINE_AA,
                )
        for index in camera_indices[np.argsort(cz[camera_indices])]:
            center = (int(cx[index]), int(cy[index]))
            cv2.circle(canvas, center, 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, center, 5, (0, 185, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, center, 8, (255, 255, 255), 1, cv2.LINE_AA)

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
    if show_cameras and camera_positions is not None:
        label = f"{len(points):,} points - {camera_count}/{len(camera_positions)} cameras visible"
        cv2.putText(
            canvas,
            label,
            (width - 470, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (180, 210, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def run_interactive(
    points,
    colors,
    camera_positions,
    width,
    height,
    yaw,
    pitch,
    roll,
    zoom,
    title,
    point_radius,
    show_cameras,
):
    while True:
        frame = render(
            points,
            colors,
            camera_positions,
            width,
            height,
            yaw,
            pitch,
            roll,
            zoom,
            title,
            point_radius,
            show_cameras,
        )
        cv2.imshow("A5 COLMAP reconstruction", frame)
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
        elif key == ord("z"):
            roll -= 5
        elif key == ord("x"):
            roll += 5
        elif key in (ord("+"), ord("=")):
            zoom *= 1.1
        elif key == ord("-"):
            zoom /= 1.1
    cv2.destroyAllWindows()


def run_gui(
    points,
    colors,
    camera_positions,
    width,
    height,
    yaw,
    pitch,
    roll,
    zoom,
    title,
    point_radius,
    show_cameras,
    screenshot,
):
    window = "A5 COLMAP viewer"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, width, height)

    def noop(_value):
        pass

    cv2.createTrackbar("yaw", window, int(round(yaw + 180)), 360, noop)
    cv2.createTrackbar("pitch", window, int(round(pitch + 180)), 360, noop)
    cv2.createTrackbar("roll", window, int(round(roll + 180)), 360, noop)
    cv2.createTrackbar("zoom x100", window, int(round(zoom * 100)), 400, noop)
    cv2.createTrackbar("point radius", window, point_radius, 4, noop)
    cv2.createTrackbar("cameras", window, 1 if show_cameras else 0, 1, noop)

    last_state = None
    frame = None
    while True:
        state = (
            cv2.getTrackbarPos("yaw", window) - 180,
            cv2.getTrackbarPos("pitch", window) - 180,
            cv2.getTrackbarPos("roll", window) - 180,
            max(1, cv2.getTrackbarPos("zoom x100", window)) / 100.0,
            max(1, cv2.getTrackbarPos("point radius", window)),
            bool(cv2.getTrackbarPos("cameras", window)),
        )
        if state != last_state:
            frame = render(
                points,
                colors,
                camera_positions,
                width,
                height,
                state[0],
                state[1],
                state[2],
                state[3],
                title,
                state[4],
                state[5],
            )
            last_state = state
        cv2.imshow(window, frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            if cv2.imwrite(str(screenshot), frame):
                print(f"Saved screenshot: {screenshot}")
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Render a colored PLY without extra 3D dependencies.")
    parser.add_argument("model", type=Path, nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--cameras", type=Path, default=DEFAULT_CAMERAS)
    parser.add_argument("--screenshot", type=Path, default=DEFAULT_SCREENSHOT)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--yaw", type=float, default=60.0)
    parser.add_argument("--pitch", type=float, default=90.0)
    parser.add_argument("--roll", type=float, default=120.0)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--title", default="A5 - COLMAP sparse reconstruction")
    parser.add_argument("--point-radius", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--hide-cameras", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    points, colors = read_ply(args.model)
    camera_positions, camera_records = read_colmap_camera_positions(args.cameras)
    points, camera_positions = normalize_scene(points, camera_positions)
    frame = render(
        points,
        colors,
        camera_positions,
        args.width,
        args.height,
        args.yaw,
        args.pitch,
        args.roll,
        args.zoom,
        args.title,
        args.point_radius,
        not args.hide_cameras,
    )
    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.screenshot), frame):
        raise RuntimeError(f"Could not write screenshot: {args.screenshot}")
    print(
        f"Rendered {len(points):,} points and {len(camera_records):,} cameras "
        f"to {args.screenshot}"
    )

    if args.gui:
        run_gui(
            points,
            colors,
            camera_positions,
            args.width,
            args.height,
            args.yaw,
            args.pitch,
            args.roll,
            args.zoom,
            args.title,
            args.point_radius,
            not args.hide_cameras,
            args.screenshot,
        )
    elif args.interactive:
        run_interactive(
            points,
            colors,
            camera_positions,
            args.width,
            args.height,
            args.yaw,
            args.pitch,
            args.roll,
            args.zoom,
            args.title,
            args.point_radius,
            not args.hide_cameras,
        )


if __name__ == "__main__":
    main()
