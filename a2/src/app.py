import cv2

from config import (
    DEFAULT_REF_SIZE,
    DEFAULT_UNIT,
    IMAGE_DIR,
    IMAGE_PATTERN,
    MAX_VIEW_HEIGHT,
    MAX_VIEW_WIDTH,
    POINT_NAMES,
    WINDOW_NAME,
)
from drawing import (
    STATUS_FOOTER_HEIGHT,
    VIEW_MODE_AUTO,
    VIEW_MODE_IMAGE,
    draw_overlay,
    render_view,
    window_to_image_point,
)
from geometry import (
    compute_target_size,
    compute_vanishing_geometry,
    format_homogeneous_point,
    normalize_h,
    parse_positive_number,
)

ENTER_KEYS = (10, 13)
BACKSPACE_KEYS = (8, 127)
DEFAULT_WINDOW_EXTRA_HEIGHT = STATUS_FOOTER_HEIGHT + 24

# Run the interactive OpenCV application loop.
def main():
    images = load_images()
    default_window_size = (MAX_VIEW_WIDTH, MAX_VIEW_HEIGHT + DEFAULT_WINDOW_EXTRA_HEIGHT)
    state = create_initial_state(images, DEFAULT_UNIT, DEFAULT_REF_SIZE, default_window_size)

    initialize_window(default_window_size, state)

    print_controls()
    print_current_image(state)

    while True:
        # Re-render every frame so window resizes and mode switches are reflected immediately.
        path, image = current_image(state)
        view, render_state = render_view(
            image,
            current_points(state),
            current_ref_size_text(state),
            state["editing_ref_size"],
            state["unit"],
            state["view_mode"],
            current_window_size(state),
        )
        state["render_state"] = render_state
        state["last_view_size"] = (view.shape[1], view.shape[0])
        cv2.imshow(WINDOW_NAME, view)

        raw_key = cv2.waitKey(20)
        if raw_key == -1:
            continue

        key = raw_key & 0xFF
        if key in (27, ord("q")):
            break
        if state["editing_ref_size"]:
            handle_ref_size_input(state, key)
            continue

        handle_key(state, key, path, image)

    cv2.destroyAllWindows()

# Load the default workshop image set.
def load_images():
    image_paths = sorted(IMAGE_DIR.glob(IMAGE_PATTERN))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR} with pattern {IMAGE_PATTERN}")

    images = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        images.append((path, image))

    return images

# Create the mutable application state shared between keyboard and mouse handlers.
def create_initial_state(images, unit, ref_size, default_window_size):
    return {
        "images": images,
        "image_index": 0,
        # Points and reference sizes are stored per image so switching images preserves work.
        "points_by_image": {path: [] for path, _ in images},
        "ref_size_text_by_image": {path: f"{ref_size:g}" for path, _ in images},
        "editing_ref_size": False,
        "replace_ref_size_on_type": False,
        "render_state": None,
        "unit": unit,
        "view_mode": VIEW_MODE_IMAGE,
        "window_size": default_window_size,
        "last_view_size": None,
    }

# Create the OpenCV window and force it back into normal decorated mode on startup.
def initialize_window(default_window_size, state):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if hasattr(cv2, "setWindowProperty") and hasattr(cv2, "WND_PROP_FULLSCREEN"):
        try:
            # macOS can keep a previous fullscreen-like window state; reset it explicitly.
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        except cv2.error:
            pass

    cv2.resizeWindow(WINDOW_NAME, *default_window_size)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, state)

# Leave reference-size edit mode and reset its temporary flags.
def clear_ref_size_editing(state):
    state["editing_ref_size"] = False
    state["replace_ref_size_on_type"] = False

# Enter reference-size edit mode and mark the first typed key as replacement.
def start_ref_size_editing(state):
    state["editing_ref_size"] = True
    state["replace_ref_size_on_type"] = True

# Handle non-text keyboard shortcuts for navigation, editing, and saving.
def handle_key(state, key, path, image):
    points = current_points(state)
    if key == ord("n"):
        next_image(state)
    elif key == ord("p"):
        previous_image(state)
    elif key == ord("z"):
        toggle_view_mode(state)
    elif key == ord("u") and points:
        removed = points.pop()
        if len(points) < 6:
            clear_ref_size_editing(state)
        print(f"Removed point: {removed}")
    elif key == ord("r"):
        reset_current_image(state)
    elif key == ord("e") and len(points) >= 6:
        start_ref_size_editing(state)
        print("Edit reference size, then press Enter to confirm.")
    elif key == ord("s"):
        save_overlay(path, image, points, current_ref_size_text(state), state["unit"])

# Clear all selected points and edit state for the current image.
def reset_current_image(state):
    current_points(state).clear()
    clear_ref_size_editing(state)
    print("Reset points for current image.")

# Print the available controls and expected click order to the console.
def print_controls():
    print(
        "\nControls:\n"
        "  left click : add table corner\n"
        "  n / p      : next / previous image\n"
        "  z          : toggle 100% / Auto-Scale view\n"
        "  u          : undo last point\n"
        "  r          : reset points for current image\n"
        "  e          : edit reference size after selecting reference object\n"
        "  s          : save overlay next to the image\n"
        "  Esc        : quit\n"
        "  q          : quit\n"
        "\nPoint order for Step 2:\n"
        "  Click the four table corners in order around the table plane.\n"
        "  Clockwise or counter-clockwise is fine.\n"
        "  Direction A: 1-2 and 3-4\n"
        "  Direction B: 2-3 and 4-1\n"
        "\nReference object:\n"
        "  After the four corners, click reference base and reference top.\n"
        "  Then type the known size/height and press Enter.\n"
        "\nTarget object:\n"
        "  After the reference size, click target base and target top.\n"
        "  The target height is estimated from the single-view geometry construction.\n"
    )

# Print the current image index, name, and resolution.
def print_current_image(state):
    path, image = current_image(state)
    height, width = image.shape[:2]
    print(
        f"Image {state['image_index'] + 1}/{len(state['images'])}: "
        f"{path.name} ({width}x{height})"
    )

# Return the currently active image tuple.
def current_image(state):
    return state["images"][state["image_index"]]

# Return the mutable point list for the current image.
def current_points(state):
    path, _ = current_image(state)
    return state["points_by_image"][path]

# Return the currently stored reference-size text for the active image.
def current_ref_size_text(state):
    path, _ = current_image(state)
    return state["ref_size_text_by_image"][path]

# Update the reference-size text for the active image.
def set_current_ref_size_text(state, text):
    path, _ = current_image(state)
    state["ref_size_text_by_image"][path] = text

# Switch to the next image and reset temporary edit state.
def next_image(state):
    state["image_index"] = (state["image_index"] + 1) % len(state["images"])
    clear_ref_size_editing(state)
    print_current_image(state)

# Switch to the previous image and reset temporary edit state.
def previous_image(state):
    state["image_index"] = (state["image_index"] - 1) % len(state["images"])
    clear_ref_size_editing(state)
    print_current_image(state)

# Toggle between the plain image view and the auto-expanded geometry view.
def toggle_view_mode(state):
    if state["view_mode"] == VIEW_MODE_IMAGE:
        state["view_mode"] = VIEW_MODE_AUTO
    else:
        state["view_mode"] = VIEW_MODE_IMAGE

# Return a stable window size without feeding the rendered image size back into scaling.
def current_window_size(state):
    default_width = MAX_VIEW_WIDTH
    default_height = MAX_VIEW_HEIGHT + DEFAULT_WINDOW_EXTRA_HEIGHT

    if not hasattr(cv2, "getWindowImageRect"):
        return state.get("window_size", (default_width, default_height))

    try:
        _, _, width, height = cv2.getWindowImageRect(WINDOW_NAME)
    except cv2.error:
        return state.get("window_size", (default_width, default_height))

    if width <= 0 or height <= 0:
        return state.get("window_size", (default_width, default_height))

    measured_size = (width, max(height, STATUS_FOOTER_HEIGHT + 25))
    last_view_size = state.get("last_view_size")

    if last_view_size is None:
        state["window_size"] = measured_size
        return measured_size

    # Ignore the size we just rendered ourselves; only keep real external window resizes.
    width_delta = abs(measured_size[0] - last_view_size[0])
    height_delta = abs(measured_size[1] - last_view_size[1])

    if width_delta > 2 or height_delta > 2:
        state["window_size"] = measured_size

    return state.get("window_size", measured_size)

# Save the current overlay visualization next to the source image.
def save_overlay(path, image, points, ref_size_text, unit):
    output_path = path.with_name(f"{path.stem}_vanishing_line.png")
    cv2.imwrite(str(output_path), draw_overlay(image, points, ref_size_text, unit))
    print(f"Saved: {output_path}")

# Compute and print the two vanishing points and the table horizon.
def print_vanishing_geometry(points):
    geometry = compute_vanishing_geometry(points)
    if geometry is None:
        print("Vanishing line could not be computed. Check the four corner points.")
        return

    v1 = normalize_h(geometry["v1"])
    v2 = normalize_h(geometry["v2"])
    horizon = geometry["horizon"]
    print(
        "Vanishing geometry:\n"
        f"  v1 = {format_homogeneous_point(v1)}\n"
        f"  v2 = {format_homogeneous_point(v2)}\n"
        f"  horizon = [{horizon[0]:.6f}, {horizon[1]:.6f}, {horizon[2]:.2f}]"
    )

# Handle text input while the user edits the reference object size.
def handle_ref_size_input(state, key):
    text = current_ref_size_text(state)

    if key in ENTER_KEYS:
        ref_size = parse_positive_number(text)
        if ref_size is None:
            print("Invalid reference size. Use a number like 28 or 28.5, then press Enter to confirm.")
            return
        set_current_ref_size_text(state, f"{ref_size:g}")
        clear_ref_size_editing(state)
        print(f"Reference size set to {ref_size:g} {state['unit']}.")
        print("Now click target base and target top.")
        return

    if key in BACKSPACE_KEYS:
        if state["replace_ref_size_on_type"]:
            # First backspace after entering edit mode clears the pre-filled default in one step.
            set_current_ref_size_text(state, "")
            state["replace_ref_size_on_type"] = False
            return
        set_current_ref_size_text(state, text[:-1])
        return

    char = chr(key)
    if char.isdigit():
        if state["replace_ref_size_on_type"]:
            set_current_ref_size_text(state, char)
            state["replace_ref_size_on_type"] = False
        else:
            set_current_ref_size_text(state, text + char)
    elif char in (".", ",") and "." not in text and "," not in text:
        if state["replace_ref_size_on_type"]:
            set_current_ref_size_text(state, "0.")
            state["replace_ref_size_on_type"] = False
        else:
            set_current_ref_size_text(state, text + char)

# Handle mouse clicks that place table, reference, and target points.
def on_mouse(event, x, y, flags, data):
    state = data

    if event == cv2.EVENT_LBUTTONDOWN:
        if state["editing_ref_size"]:
            print("Finish the reference size with Enter before selecting the target object.")
            return

        points = current_points(state)
        if len(points) >= len(POINT_NAMES):
            print("Already have table, reference, and target points. Press r to reset, u to undo, or e to edit size.")
            return

        point = window_to_image_point((x, y), state["render_state"])
        if point is None:
            # The zoomed-out auto view can show area outside the original image; clicks there are ignored.
            print("Click inside the original image area.")
            return
        points.append(point)
        point_number = len(points)
        print(f"Point {point_number} ({POINT_NAMES[point_number - 1]}): {point}")

        if point_number == 4:
            print_vanishing_geometry(points)
            print("Now click reference base and reference top.")
        elif point_number == 6:
            start_ref_size_editing(state)
            print("Type the known reference size in the image window, then press Enter to confirm.")
        elif point_number == 8:
            print_target_size(points, current_ref_size_text(state), state["unit"])

# Compute and print the estimated target object size.
def print_target_size(points, ref_size_text, unit):
    result = compute_target_size(points, ref_size_text)
    if result is None:
        print("Target size could not be computed. Check reference size and points.")
        return

    method_text = "cross-ratio"
    if result["method"] == "parallel_verticals":
        method_text = "parallel verticals"
    elif result["method"] == "pixel_ratio":
        method_text = "pixel ratio fallback"

    print(
        f"Target size: {result['target_size']:.2f} {unit} "
        f"({method_text}, target {result['target_pixels']:.1f}px, reference {result['reference_pixels']:.1f}px)"
    )
