#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(dirname "$SCRIPT_DIR")
WEB_PYTHON="$REPO_DIR/.venv/bin/python"
DA3_PYTHON="$REPO_DIR/.venv-da3/bin/python"
DA3_REPO="$SCRIPT_DIR/vendor/depth-anything-3"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "This launcher is intended for macOS." >&2
    exit 1
fi

if [ ! -x "$WEB_PYTHON" ] || [ ! -x "$DA3_PYTHON" ]; then
    echo "Missing .venv or .venv-da3. Follow the DA3 setup in a5/README.md." >&2
    exit 1
fi

if [ ! -d "$DA3_REPO/src/depth_anything_3" ]; then
    echo "Missing official Depth Anything 3 checkout at: $DA3_REPO" >&2
    exit 1
fi

if ! command -v colmap >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Install native tools first: brew install colmap ffmpeg" >&2
    exit 1
fi

export DA3_DEVICE=mps
export DA3_PYTHON
export DA3_REPO
export PYTORCH_ENABLE_MPS_FALLBACK=1

cd "$REPO_DIR"
exec "$WEB_PYTHON" a5/src/web_ui.py --host 127.0.0.1 --port 8766 --max-port-tries 1 "$@"
