import cv2
import numpy as np

from config import (
    AUTO_VIEW_MARGIN,
    COLOR_DIRECTION_A,
    COLOR_DIRECTION_A_HELPER,
    COLOR_DIRECTION_B,
    COLOR_DIRECTION_B_HELPER,
    COLOR_HORIZON,
    COLOR_POINT,
    COLOR_REFERENCE,
    COLOR_TARGET,
    COLOR_TEXT,
    COLOR_TEXT_ACTIVE,
    COLOR_TEXT_BG,
    COLOR_VANISHING_POINT,
    COLOR_VIEW_PADDING,
    EPSILON,
    MAX_VIEW_HEIGHT,
    MAX_VIEW_WIDTH,
    POINT_NAMES,
)
from geometry import (
    cartesian_point,
    compute_height_measurement_geometry,
    compute_target_size,
    compute_vanishing_geometry,
    line_from_point_and_homogeneous,
    line_from_points,
    line_rectangle_intersections,
    parse_positive_number,
    pixel_distance,
)
from text_rendering import draw_text_items, text_metrics


VIEW_MODE_IMAGE = "image"
VIEW_MODE_AUTO = "auto"
STATUS_FOOTER_HEIGHT = 74
FRAME_COLOR = (80, 80, 80)
STATUS_FIELD_HEIGHT = 22

# Render the full interactive view and return it together with click-mapping data.
def render_view(image, points, ref_size_text, editing_ref_size, unit, view_mode, available_size=None):
    # The viewport describes how image/world coordinates are projected into the current window.
    viewport = build_viewport(image, points, view_mode, available_size)
    canvas = render_canvas(image, points, ref_size_text, unit, viewport)
    view = add_status_header(canvas, STATUS_FOOTER_HEIGHT)
    draw_status_box(view, points, ref_size_text, editing_ref_size, unit, view_mode)

    render_state = {
        "scale": viewport["scale"],
        "origin": viewport["origin"],
        "canvas_offset": (
            viewport["canvas_offset"][0],
            viewport["canvas_offset"][1] + STATUS_FOOTER_HEIGHT,
        ),
        "image_size": (image.shape[1], image.shape[0]),
    }
    return view, render_state

# Render an overlay directly on the original image for saving to disk.
def draw_overlay(image, points, ref_size_text, unit):
    canvas = image.copy()
    viewport = full_image_viewport(image)
    draw_image_frame(canvas, (0, 0), image.shape[1], image.shape[0])
    draw_overlay_elements(canvas, points, ref_size_text, unit, viewport)
    return canvas

# Build the viewport that maps world coordinates into the current canvas.
def build_viewport(image, points, view_mode, available_size=None):
    image_height, image_width = image.shape[:2]
    # In auto mode the world rectangle expands until the vanishing points fit into the rendered view.
    world_rect = world_bounds(image_width, image_height, points, view_mode)
    world_min_x, world_min_y, world_max_x, world_max_y = world_rect
    world_width = max(1.0, world_max_x - world_min_x)
    world_height = max(1.0, world_max_y - world_min_y)

    available_width, available_height = viewport_available_size(available_size)
    scale = min(available_width / world_width, available_height / world_height, 1.0)

    projected_width = max(1, int(round(world_width * scale)))
    projected_height = max(1, int(round(world_height * scale)))
    canvas_width = max(1, available_width)
    canvas_height = max(1, available_height)
    canvas_offset = (
        max(0, (canvas_width - projected_width) // 2),
        max(0, (canvas_height - projected_height) // 2),
    )
    style_scale = 1.0 if view_mode == VIEW_MODE_IMAGE else 0.8

    return {
        "canvas_size": (canvas_width, canvas_height),
        "world_rect": world_rect,
        "scale": scale,
        "origin": (world_min_x, world_min_y),
        "canvas_offset": canvas_offset,
        "compact_labels": view_mode == VIEW_MODE_AUTO,
        "style_scale": style_scale,
    }

# Create a viewport that matches the original image exactly.
def full_image_viewport(image):
    return {
        "canvas_size": (image.shape[1], image.shape[0]),
        "world_rect": (0.0, 0.0, float(image.shape[1] - 1), float(image.shape[0] - 1)),
        "scale": 1.0,
        "origin": (0.0, 0.0),
        "canvas_offset": (0, 0),
        "compact_labels": False,
        "style_scale": 1.0,
    }

# Compute the drawable size after removing outer padding and footer space.
def viewport_available_size(available_size):
    if available_size is None:
        total_width = MAX_VIEW_WIDTH
        total_height = MAX_VIEW_HEIGHT
    else:
        total_width, total_height = available_size

    usable_width = max(1, int(round(total_width)))
    usable_height = max(1, int(round(total_height)) - STATUS_FOOTER_HEIGHT)
    return usable_width, usable_height

# Return the world rectangle that should be visible for the chosen view mode.
def world_bounds(image_width, image_height, points, view_mode):
    image_min_x = 0.0
    image_min_y = 0.0
    image_max_x = float(image_width - 1)
    image_max_y = float(image_height - 1)

    if view_mode != VIEW_MODE_AUTO or len(points) < 4:
        return image_min_x, image_min_y, image_max_x, image_max_y

    xs = [image_min_x, image_max_x]
    ys = [image_min_y, image_max_y]

    geometry = compute_vanishing_geometry(points)
    if geometry is not None:
        for key in ("v1", "v2"):
            point_h = geometry[key]
            point = cartesian_point(point_h)
            if point is None:
                continue
            xs.append(float(point[0]))
            ys.append(float(point[1]))

    measurement_geometry = compute_height_measurement_geometry(points)
    if measurement_geometry is not None:
        for key in ("auxiliary_vanishing_point", "transferred_target_top"):
            point = cartesian_point(measurement_geometry[key])
            if point is None:
                continue
            xs.append(float(point[0]))
            ys.append(float(point[1]))

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    margin = max(AUTO_VIEW_MARGIN, 0.05 * max(image_width, image_height))
    return min_x - margin, min_y - margin, max_x + margin, max_y + margin

# Render the image and all overlays into the projected canvas.
def render_canvas(image, points, ref_size_text, unit, viewport):
    canvas_width, canvas_height = viewport["canvas_size"]
    canvas = np.full((canvas_height, canvas_width, 3), COLOR_VIEW_PADDING, dtype=image.dtype)
    # First place the source image into the projected viewport, then draw all geometric overlays on top.
    draw_projected_image(canvas, image, viewport)
    draw_overlay_elements(canvas, points, ref_size_text, unit, viewport)
    return canvas

# Project the source image into the current viewport rectangle.
def draw_projected_image(canvas, image, viewport):
    image_height, image_width = image.shape[:2]
    top_left = project_point((0.0, 0.0), viewport)
    bottom_right = project_point((float(image_width), float(image_height)), viewport)

    x0 = max(0, min(canvas.shape[1] - 1, top_left[0]))
    y0 = max(0, min(canvas.shape[0] - 1, top_left[1]))
    x1 = max(x0 + 1, min(canvas.shape[1], bottom_right[0]))
    y1 = max(y0 + 1, min(canvas.shape[0], bottom_right[1]))

    resized = cv2.resize(image, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    canvas[y0:y1, x0:x1] = resized
    draw_image_frame(canvas, (x0, y0), x1 - x0, y1 - y0)

# Draw a border around the projected image region.
def draw_image_frame(image, origin, width, height):
    x, y = origin
    cv2.rectangle(image, (x - 1, y - 1), (x + width, y + height), FRAME_COLOR, 1)

# Draw all geometric overlays, markers, and labels for the current scene.
def draw_overlay_elements(image, points, ref_size_text, unit, viewport):
    text_items = []

    draw_table_edges(image, points, viewport)
    draw_table_helper_lines(image, points, viewport)

    geometry = compute_vanishing_geometry(points)
    if geometry is not None:
        draw_projected_infinite_line(
            image,
            geometry["horizon"],
            COLOR_HORIZON,
            scaled_size(2, viewport),
            viewport,
        )
        draw_vanishing_points(image, geometry, text_items, viewport)
        draw_object_helper_lines(image, points, geometry, viewport)
        draw_height_measurement_guides(image, points, text_items, viewport)

    for index, point in enumerate(points[:4], start=1):
        draw_point(image, point, index, text_items, viewport)

    draw_reference_object(image, points, ref_size_text, unit, text_items, viewport)
    draw_target_object(image, points, ref_size_text, unit, text_items, viewport)
    draw_text_items(image, text_items)

# Draw the selected table quadrilateral edges.
def draw_table_edges(image, points, viewport):
    thickness = scaled_size(2, viewport)
    if len(points) >= 2:
        cv2.line(image, project_point(points[0], viewport), project_point(points[1], viewport), COLOR_DIRECTION_A, thickness)
    if len(points) >= 3:
        cv2.line(image, project_point(points[1], viewport), project_point(points[2], viewport), COLOR_DIRECTION_B, thickness)
    if len(points) >= 4:
        cv2.line(image, project_point(points[2], viewport), project_point(points[3], viewport), COLOR_DIRECTION_A, thickness)
        cv2.line(image, project_point(points[3], viewport), project_point(points[0], viewport), COLOR_DIRECTION_B, thickness)

# Draw the infinite helper lines used to construct the vanishing points.
def draw_table_helper_lines(image, points, viewport):
    thickness = scaled_size(2, viewport)
    if len(points) >= 2:
        draw_projected_infinite_line(image, line_from_points(points[0], points[1]), COLOR_DIRECTION_A_HELPER, thickness, viewport)
    if len(points) >= 3:
        draw_projected_infinite_line(image, line_from_points(points[1], points[2]), COLOR_DIRECTION_B_HELPER, thickness, viewport)
    if len(points) >= 4:
        draw_projected_infinite_line(image, line_from_points(points[2], points[3]), COLOR_DIRECTION_A_HELPER, thickness, viewport)
        draw_projected_infinite_line(image, line_from_points(points[3], points[0]), COLOR_DIRECTION_B_HELPER, thickness, viewport)

# Draw visible vanishing points and their labels.
def draw_vanishing_points(image, geometry, text_items, viewport):
    outer_radius = scaled_size(4, viewport)
    inner_radius = scaled_size(1, viewport)
    offset = scaled_size(5, viewport)
    font_size = scaled_size(10, viewport)
    thickness = scaled_size(1, viewport)

    for label, key in (("v1", "v1"), ("v2", "v2")):
        draw_named_homogeneous_point(
            image,
            geometry[key],
            label,
            COLOR_VANISHING_POINT,
            outer_radius,
            thickness,
            text_items,
            viewport,
            fill_radius=inner_radius,
            font_size=font_size,
            offset=offset,
        )

# Draw helper lines from the vanishing points to the object foot points on the table plane.
def draw_object_helper_lines(image, points, geometry, viewport):
    if len(points) >= 5:
        draw_ground_guides_for_point(image, points[4], geometry, COLOR_REFERENCE, viewport)
    if len(points) >= 7:
        draw_ground_guides_for_point(image, points[6], geometry, COLOR_TARGET, viewport)


# Draw both ground-plane guide lines through one selected foot point.
def draw_ground_guides_for_point(image, point, geometry, color, viewport):
    thickness = scaled_size(1, viewport)
    for key in ("v1", "v2"):
        line = line_from_point_and_homogeneous(point, geometry[key])
        if line is not None:
            draw_projected_infinite_line(image, line, color, thickness, viewport)

# Draw the slide-like height-transfer construction between target and reference object.
def draw_height_measurement_guides(image, points, text_items, viewport):
    geometry = compute_height_measurement_geometry(points)
    if geometry is None:
        return

    helper_thickness = scaled_size(1, viewport)
    main_thickness = scaled_size(2, viewport)
    marker_radius = scaled_size(4, viewport)
    marker_fill = scaled_size(1, viewport)
    offset = scaled_size(6, viewport)
    font_size = scaled_size(10, viewport)

    draw_projected_infinite_line(image, geometry["base_connection"], COLOR_HORIZON, helper_thickness, viewport)
    draw_projected_infinite_line(image, geometry["transfer_line"], COLOR_HORIZON, main_thickness, viewport)
    draw_projected_infinite_line(image, geometry["reference_vertical"], COLOR_REFERENCE, helper_thickness, viewport)
    draw_projected_infinite_line(image, geometry["target_vertical"], COLOR_TARGET, helper_thickness, viewport)

    draw_named_homogeneous_point(
        image,
        geometry["auxiliary_vanishing_point"],
        "v",
        COLOR_POINT,
        marker_radius,
        main_thickness,
        text_items,
        viewport,
        fill_radius=marker_fill,
        font_size=font_size,
        offset=offset,
    )
    draw_named_homogeneous_point(
        image,
        geometry["transferred_target_top"],
        "t",
        COLOR_POINT,
        marker_radius,
        main_thickness,
        text_items,
        viewport,
        fill_radius=marker_fill,
        font_size=font_size,
        offset=offset,
    )
    draw_named_homogeneous_point(
        image,
        geometry["vertical_vanishing_point"],
        "vz",
        COLOR_POINT,
        marker_radius,
        main_thickness,
        text_items,
        viewport,
        fill_radius=marker_fill,
        font_size=font_size,
        offset=offset,
    )

# Draw and label one finite homogeneous point if it lies inside the current canvas.
def draw_named_homogeneous_point(
    image,
    point_h,
    label,
    color,
    radius,
    thickness,
    text_items,
    viewport,
    fill_radius=None,
    font_size=None,
    offset=None,
):
    point = cartesian_point(point_h)
    if point is None:
        return

    projected = project_point(point, viewport)
    if not is_inside_canvas(projected, image):
        return

    if font_size is None:
        font_size = scaled_size(10, viewport)
    if offset is None:
        offset = scaled_size(5, viewport)

    draw_marker(image, projected, color, radius, thickness, fill_radius=fill_radius)
    text_items.append((label, (projected[0] + offset, projected[1] - offset), font_size, color, thickness))

# Draw one numbered table corner marker.
def draw_point(image, point, index, text_items, viewport):
    projected = project_point(point, viewport)
    draw_labeled_marker(image, projected, str(index), COLOR_POINT, viewport, text_items)

# Draw the reference segment, its markers, and its optional size label.
def draw_reference_object(image, points, ref_size_text, unit, text_items, viewport):
    if len(points) < 5:
        return

    for index, point in enumerate(points[4:6], start=1):
        projected = project_point(point, viewport)
        draw_labeled_marker(image, projected, f"R{index}", COLOR_REFERENCE, viewport, text_items)

    if len(points) < 6:
        return

    ref_base, ref_top = project_segment(points[4], points[5], viewport)
    draw_segment(image, ref_base, ref_top, COLOR_REFERENCE, viewport)

    ref_size = parse_positive_number(ref_size_text)
    if ref_size is None:
        return

    if not viewport["compact_labels"]:
        label = f"Reference: {ref_size:g} {unit}"
        label_pos = (
            int(round((ref_base[0] + ref_top[0]) / 2)) + scaled_size(12, viewport),
            int(round((ref_base[1] + ref_top[1]) / 2)),
        )
        draw_label(image, label, label_pos, COLOR_REFERENCE, text_items, viewport)

# Draw the target segment, its markers, and its optional result label.
def draw_target_object(image, points, ref_size_text, unit, text_items, viewport):
    if len(points) < 7:
        return

    for index, point in enumerate(points[6:8], start=1):
        projected = project_point(point, viewport)
        draw_labeled_marker(image, projected, f"T{index}", COLOR_TARGET, viewport, text_items)

    if len(points) < 8:
        return

    target_base, target_top = project_segment(points[6], points[7], viewport)
    draw_segment(image, target_base, target_top, COLOR_TARGET, viewport)

    result = compute_target_size(points, ref_size_text)
    if not viewport["compact_labels"]:
        if result is None:
            label = f"Target: {pixel_distance(points[6], points[7]):.1f} px"
        else:
            label = f"Target: {result['target_size']:.2f} {unit}"

        label_pos = (
            int(round((target_base[0] + target_top[0]) / 2)) + scaled_size(12, viewport),
            int(round((target_base[1] + target_top[1]) / 2)),
        )
        draw_label(image, label, label_pos, COLOR_TARGET, text_items, viewport)

# Queue a boxed text label while keeping it inside the image bounds.
def draw_label(image, text, position, color, text_items, viewport):
    font_size = scaled_size(15, viewport)
    thickness = scaled_size(1, viewport)
    pad_x = scaled_size(3, viewport)
    pad_y = scaled_size(4, viewport)
    margin = scaled_size(5, viewport)

    x, y = position
    (text_width, text_height), baseline = text_metrics(text, font_size, thickness)
    x = max(6, min(image.shape[1] - text_width - scaled_size(12, viewport), x))
    y = max(text_height + margin, min(image.shape[0] - baseline - margin, y))

    cv2.rectangle(
        image,
        (x - pad_x, y - text_height - pad_y),
        (x + text_width + pad_x, y + baseline + pad_x),
        (0, 0, 0),
        -1,
    )
    text_items.append((text, (x, y), font_size, color, thickness))

# Draw a filled marker together with its short text label.
def draw_labeled_marker(image, point, label, color, viewport, text_items):
    radius = scaled_size(3, viewport)
    offset = scaled_size(5, viewport)
    font_size = scaled_size(11, viewport)
    thickness = scaled_size(1, viewport)
    draw_marker(image, point, color, radius, -1)
    text_items.append((label, (point[0] + offset, point[1] - offset), font_size, color, thickness))

# Draw a circular marker, optionally with a filled center.
def draw_marker(image, point, color, radius, thickness, fill_radius=None):
    cv2.circle(image, point, radius, color, thickness)
    if fill_radius is not None:
        cv2.circle(image, point, fill_radius, color, -1)

# Project both endpoints of a segment into viewport coordinates.
def project_segment(start_point, end_point, viewport):
    return project_point(start_point, viewport), project_point(end_point, viewport)

# Draw a projected line segment with viewport-aware thickness.
def draw_segment(image, start_point, end_point, color, viewport):
    cv2.line(image, start_point, end_point, color, scaled_size(3, viewport))

# Draw the status bar in the header area above the image.
def draw_status_box(image, points, ref_size_text, editing_ref_size, unit, view_mode):
    status, field = status_text(points, ref_size_text, editing_ref_size, unit)
    mode_text = "View: 100%" if view_mode == VIEW_MODE_IMAGE else "View: Auto-Scale"
    panel_left, panel_top, panel_right, panel_bottom = status_panel_rect(image.shape[1], field is not None)
    cv2.rectangle(image, (panel_left, panel_top), (panel_right, panel_bottom), COLOR_TEXT_BG, -1)
    cv2.rectangle(
        image,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        COLOR_TEXT_ACTIVE if editing_ref_size else COLOR_TEXT,
        1,
    )
    text_items = [
        (status, (panel_left + 12, panel_top + 22), 16, COLOR_TEXT, 1),
        (mode_text, (panel_left + 12, panel_top + 42), 13, COLOR_TEXT, 1),
    ]

    if field is not None:
        field_left, field_top, field_right, field_bottom = status_field_rect(panel_left, panel_top, panel_right)
        cv2.rectangle(image, (field_left, field_top), (field_right, field_bottom), (245, 245, 245), -1)
        cv2.rectangle(image, (field_left, field_top), (field_right, field_bottom), COLOR_TEXT_ACTIVE, 1)
        text_items.append((field, (field_left + 8, field_top + 15), 12, (0, 0, 0), 1))

    draw_text_items(image, text_items)

# Return the status message and optional footer field text for the current step.
def status_text(points, ref_size_text, editing_ref_size, unit):
    if len(points) < 4:
        return f"Click table corner {len(points) + 1}/4", None
    if len(points) < 6:
        return f"Click {POINT_NAMES[len(points)]}", None
    if editing_ref_size:
        return "Reference size: press Enter to confirm", f"{ref_size_text}| {unit}"
    if len(points) < 8:
        return f"Click {POINT_NAMES[len(points)]}", f"ref: {ref_size_text} {unit}"

    result = compute_target_size(points, ref_size_text)
    if result is None:
        return "Target distance", "invalid reference size"

    return f"Target size: {result['target_size']:.2f} {unit}", None

# Clip an infinite line to the viewport rectangle and draw the visible part.
def draw_projected_infinite_line(image, line, color, thickness, viewport):
    # Intersect the homogeneous line with the current viewport rectangle so we can draw it on screen.
    world_min_x, world_min_y, world_max_x, world_max_y = viewport["world_rect"]
    intersections = line_rectangle_intersections(line, world_min_x, world_min_y, world_max_x, world_max_y)
    if len(intersections) < 2:
        return

    start = project_point(intersections[0], viewport)
    end = project_point(intersections[1], viewport)
    cv2.line(image, start, end, color, thickness)

# Scale a visual size according to the current viewport style settings.
def scaled_size(base_size, viewport):
    return max(1, int(round(base_size * viewport["style_scale"])))

# Compute the header status panel rectangle.
def status_panel_rect(image_width, has_field):
    panel_left = 0
    panel_top = 0
    panel_right = image_width - 1
    panel_bottom = STATUS_FOOTER_HEIGHT - 1
    return panel_left, panel_top, panel_right, panel_bottom

# Compute the input field rectangle inside the header bar.
def status_field_rect(panel_left, panel_top, panel_right):
    field_width = min(320, max(180, (panel_right - panel_left) // 3))
    field_right = panel_right - 16
    field_left = field_right - field_width
    field_top = panel_top + 14
    field_bottom = field_top + STATUS_FIELD_HEIGHT
    return field_left, field_top, field_right, field_bottom

# Add a solid status header above the rendered image.
def add_status_header(image, header_height):
    height, width = image.shape[:2]
    view = np.full(
        (height + header_height, width, 3),
        COLOR_VIEW_PADDING,
        dtype=image.dtype,
    )
    view[header_height:, :width] = image
    cv2.rectangle(view, (0, 0), (width - 1, header_height - 1), FRAME_COLOR, 1)
    cv2.rectangle(view, (0, header_height), (width - 1, height + header_height - 1), FRAME_COLOR, 1)
    return view

# Project one world-space point into canvas pixel coordinates.
def project_point(point, viewport):
    origin_x, origin_y = viewport["origin"]
    offset_x, offset_y = viewport["canvas_offset"]
    scale = viewport["scale"]
    # World coordinates are shifted by the viewport origin and then scaled into canvas pixels.
    x = int(round((float(point[0]) - origin_x) * scale)) + offset_x
    y = int(round((float(point[1]) - origin_y) * scale)) + offset_y
    return x, y

# Check whether a projected point lies inside the current canvas.
def is_inside_canvas(point, image):
    return 0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0]

# Keep the helper for compatibility even though the GUI now sits over the image.
def add_footer(image, footer_height=0):
    if footer_height <= 0:
        return image

    height, width = image.shape[:2]
    view = np.full(
        (height + footer_height, width, 3),
        COLOR_VIEW_PADDING,
        dtype=image.dtype,
    )
    view[:height, :width] = image
    cv2.rectangle(view, (0, 0), (width - 1, height - 1), FRAME_COLOR, 1)
    return view

# Map a window click back into original image coordinates if it hits the image area.
def window_to_image_point(point, render_state):
    if render_state is None:
        return None

    scale = render_state["scale"]
    origin_x, origin_y = render_state["origin"]
    offset_x, offset_y = render_state["canvas_offset"]
    image_width, image_height = render_state["image_size"]

    # Convert from window pixels back into original image coordinates for point selection.
    world_x = (point[0] - offset_x) / scale + origin_x
    world_y = (point[1] - offset_y) / scale + origin_y

    if not (0.0 <= world_x < image_width and 0.0 <= world_y < image_height):
        return None

    image_x = max(0, min(image_width - 1, int(round(world_x))))
    image_y = max(0, min(image_height - 1, int(round(world_y))))
    return image_x, image_y
