from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR.parent / "img"
IMAGE_PATTERN = "table_bottle*.jpeg"
WINDOW_NAME = "scene"
DEFAULT_REF_SIZE = 28.0
DEFAULT_UNIT = "cm"

MAX_VIEW_WIDTH = 1200
MAX_VIEW_HEIGHT = 850
EPSILON = 1e-9
AUTO_VIEW_MARGIN = 80.0

COLOR_POINT = (0, 0, 255)
COLOR_DIRECTION_A = (0, 180, 0)
COLOR_DIRECTION_B = (255, 120, 0)
COLOR_DIRECTION_A_HELPER = (120, 220, 120)
COLOR_DIRECTION_B_HELPER = (255, 190, 120)
COLOR_HORIZON = (255, 0, 255)
COLOR_VANISHING_POINT = (255, 255, 255)
COLOR_REFERENCE = (0, 220, 0)
COLOR_TEXT_BG = (35, 35, 35)
COLOR_TEXT = (245, 245, 245)
COLOR_TEXT_ACTIVE = (0, 220, 0)
COLOR_TARGET = (0, 165, 255)
COLOR_VIEW_PADDING = (22, 22, 22)

POINT_NAMES = [
    "corner 1",
    "corner 2",
    "corner 3",
    "corner 4",
    "reference base",
    "reference top",
    "target base",
    "target top",
]
