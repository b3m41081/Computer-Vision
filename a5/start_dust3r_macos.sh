#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(dirname "$SCRIPT_DIR")
PYTHON="$SCRIPT_DIR/.venv-dust3r/bin/python"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "This launcher is intended for macOS." >&2
    exit 1
fi

if [ ! -x "$PYTHON" ]; then
    echo "Missing a5/.venv-dust3r. Install DUSt3R dependencies first." >&2
    exit 1
fi

cd "$REPO_DIR"
exec "$PYTHON" a5/src/run_dust3r_local.py "$@"
