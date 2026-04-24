import numpy as np

from config import EPSILON

# Convert a 2D image point into homogeneous coordinates.
def to_h(point):
    return np.array([float(point[0]), float(point[1]), 1.0])

# Construct a homogeneous line through two image points.
def line_from_points(point_a, point_b):
    line = np.cross(to_h(point_a), to_h(point_b))
    norm = np.linalg.norm(line[:2])
    if norm < EPSILON:
        return line
    return line / norm

# Construct a homogeneous line through one image point and one homogeneous point.
def line_from_point_and_homogeneous(point, point_h):
    line = np.cross(to_h(point), point_h)
    norm = np.linalg.norm(line[:2])
    if norm < EPSILON:
        return None
    return line / norm

# Intersect two homogeneous lines and return the homogeneous point.
def intersect_lines(line_a, line_b):
    return np.cross(line_a, line_b)

# Normalize a homogeneous point to w=1 when possible.
def normalize_h(point):
    if abs(point[2]) < EPSILON:
        return point
    return point / point[2]

# Convert a finite homogeneous point into a 2D tuple.
def cartesian_point(point):
    if abs(point[2]) < EPSILON:
        return None
    normalized = normalize_h(point)
    return float(normalized[0]), float(normalized[1])

# Check whether a vector is numerically too small to be usable.
def is_degenerate(value):
    return np.linalg.norm(value) < EPSILON

# Compute both table-plane vanishing points and the horizon line.
def compute_vanishing_geometry(points):
    if len(points) < 4:
        return None

    line_a1 = line_from_points(points[0], points[1])
    line_a2 = line_from_points(points[2], points[3])
    line_b1 = line_from_points(points[1], points[2])
    line_b2 = line_from_points(points[3], points[0])

    v1 = intersect_lines(line_a1, line_a2)
    v2 = intersect_lines(line_b1, line_b2)
    horizon = np.cross(v1, v2)

    if is_degenerate(v1) or is_degenerate(v2):
        return None
    if np.linalg.norm(horizon[:2]) < EPSILON:
        return None

    horizon = horizon / np.linalg.norm(horizon[:2])
    return {"v1": v1, "v2": v2, "horizon": horizon}

# Intersect a homogeneous line with an axis-aligned rectangle.
def line_rectangle_intersections(line, x_min, y_min, x_max, y_max):
    a, b, c = line
    intersections = []

    if abs(b) > EPSILON:
        for x in (x_min, x_max):
            y = -(a * x + c) / b
            if y_min <= y <= y_max:
                intersections.append((int(round(x)), int(round(y))))

    if abs(a) > EPSILON:
        for y in (y_min, y_max):
            x = -(b * y + c) / a
            if x_min <= x <= x_max:
                intersections.append((int(round(x)), int(round(y))))

    return list(dict.fromkeys(intersections))

# Compute the Euclidean distance between two image points.
def pixel_distance(point_a, point_b):
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return float(np.hypot(dx, dy))

# Parse a float from user text, accepting comma or dot decimals.
def parse_positive_number(text):
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None

# Compute the geometric construction used for single-view height transfer.
def compute_height_measurement_geometry(points):
    if len(points) < 8:
        return None

    table_geometry = compute_vanishing_geometry(points)
    if table_geometry is None:
        return None

    reference_base, reference_top = points[4], points[5]
    target_base, target_top = points[6], points[7]

    reference_vertical = line_from_points(reference_base, reference_top)
    target_vertical = line_from_points(target_base, target_top)
    base_connection = line_from_points(reference_base, target_base)

    auxiliary_vanishing_point = intersect_lines(base_connection, table_geometry["horizon"])
    if is_degenerate(auxiliary_vanishing_point):
        return None

    transfer_line = line_from_point_and_homogeneous(target_top, auxiliary_vanishing_point)
    if transfer_line is None:
        return None

    transferred_target_top = intersect_lines(transfer_line, reference_vertical)
    if is_degenerate(transferred_target_top):
        return None

    vertical_vanishing_point = intersect_lines(reference_vertical, target_vertical)
    if is_degenerate(vertical_vanishing_point):
        return None

    return {
        "table_geometry": table_geometry,
        "reference_vertical": reference_vertical,
        "target_vertical": target_vertical,
        "base_connection": base_connection,
        "auxiliary_vanishing_point": auxiliary_vanishing_point,
        "transfer_line": transfer_line,
        "transferred_target_top": transferred_target_top,
        "vertical_vanishing_point": vertical_vanishing_point,
    }

# Estimate the target size using the height-transfer construction from the slide.
def compute_target_size(points, ref_size_text):
    if len(points) < 8:
        return None

    ref_size = parse_positive_number(ref_size_text)
    if ref_size is None:
        return None

    ref_pixels = pixel_distance(points[4], points[5])
    target_pixels = pixel_distance(points[6], points[7])
    if ref_pixels < EPSILON:
        return None

    geometry = compute_height_measurement_geometry(points)
    if geometry is None:
        target_size = target_pixels / ref_pixels * ref_size
        return {
            "target_size": target_size,
            "target_pixels": target_pixels,
            "reference_pixels": ref_pixels,
            "transfer_pixels": None,
            "method": "pixel_ratio",
        }

    transferred_target_top = cartesian_point(geometry["transferred_target_top"])
    if transferred_target_top is None:
        return None

    transfer_pixels = pixel_distance(points[4], transferred_target_top)
    if transfer_pixels < EPSILON:
        return None

    vertical_vanishing_point = geometry["vertical_vanishing_point"]
    if abs(vertical_vanishing_point[2]) < EPSILON:
        target_size = transfer_pixels / ref_pixels * ref_size
        method = "parallel_verticals"
    else:
        vertical_point = cartesian_point(vertical_vanishing_point)
        if vertical_point is None:
            return None

        reference_to_vz = pixel_distance(vertical_point, points[5])
        transfer_to_vz = pixel_distance(vertical_point, transferred_target_top)
        if reference_to_vz < EPSILON or transfer_to_vz < EPSILON:
            return None

        target_size = transfer_pixels * reference_to_vz / (ref_pixels * transfer_to_vz) * ref_size
        method = "cross_ratio"

    return {
        "target_size": target_size,
        "target_pixels": target_pixels,
        "reference_pixels": ref_pixels,
        "transfer_pixels": transfer_pixels,
        "method": method,
    }

# Format a homogeneous point for readable console output.
def format_homogeneous_point(point):
    if abs(point[2]) < EPSILON:
        return f"at infinity [{point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}]"
    return f"({point[0]:.1f}, {point[1]:.1f})"
