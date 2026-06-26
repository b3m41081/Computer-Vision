#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(dirname "$SCRIPT_DIR")
PYTHON="$REPO_DIR/.venv/bin/python"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "This launcher is intended for macOS." >&2
    exit 1
fi

if [ ! -x "$PYTHON" ]; then
    echo "Missing .venv. Follow the A5 COLMAP setup in a5/README.md." >&2
    exit 1
fi

if ! command -v colmap >/dev/null 2>&1; then
    echo "Install native COLMAP first: brew install colmap" >&2
    exit 1
fi

cd "$REPO_DIR"
exec "$PYTHON" a5/src/run_colmap_pipeline.py "$@"
