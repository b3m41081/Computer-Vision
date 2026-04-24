import cv2

# Draw a list of queued text items onto the image.
def draw_text_items(image, text_items):
    for text, position, font_size, color, thickness in text_items:
        draw_text(image, text, position, font_size, color, thickness)

# Draw one text string with the shared OpenCV text settings.
def draw_text(image, text, position, font_size, color, thickness):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size / 30.0,
        color,
        max(1, thickness),
        cv2.LINE_AA,
    )

# Return the pixel size and baseline for a text string.
def text_metrics(text, font_size, thickness=1):
    return cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size / 30.0,
        max(1, thickness),
    )
