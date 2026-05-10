# A2: Single-View Height Measurement

Example with the geometric construction drawn on top:
![Overlay](img/table_bottle_vanishing_line.png)

## Run

From the project root:

```bash
.venv/bin/python a2/src/main.py
```

## Requirements

- Python 3
- `opencv-python`
- `numpy`

Setup from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After that, the application can be started in the same terminal:

```bash
.venv/bin/python a2/src/main.py
```

## Usage

Click order:

1. `corner 1`
2. `corner 2`
3. `corner 3`
4. `corner 4`
5. `reference base`
6. `reference top`
7. after point 6, enter the known reference height
8. confirm with `Enter`
9. then click the target base and target top
10. `target base`
11. `target top`

Keyboard shortcuts:

- `n` next image
- `p` previous image
- `z` toggle between `100%` and `Auto-Scale`
- `u` undo the last point
- `r` clear points of the current image
- `e` enter the reference height again
- `s` save the overlay image
- `Esc` or `q` quit

The `Auto-Scale` mode enlarges the visible world area so that important
geometric constructions are easier to inspect.

## Folder Structure

```text
a2/
├── img/
│   ├── slide61_ws02.png
│   ├── table_bottle01.jpeg
│   ├── table_bottle02.jpeg
│   ├── table_bottle03.jpeg
│   └── *_vanishing_line.png
└── src/
    ├── main.py
    ├── app.py
    ├── config.py
    ├── drawing.py
    ├── geometry.py
    └── text_rendering.py
```

Short file roles:

- `main.py`: entry point
- `app.py`: GUI loop, input handling, image switching, keyboard and mouse
- `config.py`: constants, colors, image paths
- `drawing.py`: overlay and GUI rendering
- `geometry.py`: vanishing-point, horizon, and height geometry
- `text_rendering.py`: small OpenCV text helpers

## Adding Images

By default, all images matching this pattern are loaded:

```text
a2/img/table_bottle*.jpeg
```

Examples:

- `table_bottle04.jpeg`
- `table_bottle_test.jpeg`

## Output

Pressing `s` saves an overlay next to the original image, for example:

```text
table_bottle01_vanishing_line.png
```

This image contains the current geometric construction and the measured objects.

## Results

The three overlays document the measurement including the construction and the
computed cup height.

| Image | Estimated cup height | Overlay |
| --- | ---: | --- |
| `table_bottle01.jpeg` | `11.29 cm` | [open](img/table_bottle01_vanishing_line.png) |
| `table_bottle02.jpeg` | `10.29 cm` | [open](img/table_bottle02_vanishing_line.png) |
| `table_bottle03.jpeg` | `10.07 cm` | [open](img/table_bottle03_vanishing_line.png) |

The mean value of the three measurements is `10.55 cm`.
