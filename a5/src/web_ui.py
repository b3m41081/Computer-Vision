import argparse
import json
import mimetypes
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BASE_DIR.parent
VIDEO_DIR = BASE_DIR / "video"
MANIFEST = BASE_DIR / "data" / "scene" / "video_frames.json"
IMAGE_DIR = BASE_DIR / "data" / "scene" / "images"
MODEL = BASE_DIR / "colmap" / "sparse_points.ply"
SPARSE_MODEL = BASE_DIR / "colmap" / "sparse" / "0"
SCREENSHOT = BASE_DIR / "img" / "colmap_sparse.png"
DA3_MODEL = BASE_DIR / "da3" / "points.ply"
DA3_METADATA = BASE_DIR / "da3" / "metadata.json"
DA3_SCREENSHOT = BASE_DIR / "img" / "da3_reconstruction.png"
WEB_DIR = Path(__file__).with_name("web")
STATIC_DIR = WEB_DIR / "static"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def bounded_number(payload, name, default, minimum, maximum, number_type=float):
    value = number_type(payload.get(name, default))
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def parse_ply_vertex_count(path):
    if not path.is_file():
        return None
    with path.open("rb") as file:
        for raw_line in file:
            line = raw_line.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def model_statistics():
    if not SPARSE_MODEL.is_dir() or shutil.which("colmap") is None:
        return {}
    result = subprocess.run(
        ["colmap", "model_analyzer", "--path", str(SPARSE_MODEL)],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        timeout=20,
    )
    output = result.stdout + result.stderr
    patterns = {
        "registered_images": r"Registered images:\s+(\d+)",
        "points": r"Points:\s+(\d+)",
        "observations": r"Observations:\s+(\d+)",
        "mean_track_length": r"Mean track length:\s+([0-9.]+)",
        "mean_reprojection_error": r"Mean reprojection error:\s+([0-9.]+)px",
    }
    statistics = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            value = match.group(1)
            statistics[name] = float(value) if "." in value else int(value)
    return statistics


def current_results():
    results = {
        "image_count": len(list(IMAGE_DIR.glob("frame_*.jpg"))) if IMAGE_DIR.is_dir() else 0,
        "model_exists": MODEL.is_file(),
        "model_size": MODEL.stat().st_size if MODEL.is_file() else 0,
        "screenshot_exists": SCREENSHOT.is_file(),
        "screenshot_version": int(SCREENSHOT.stat().st_mtime_ns) if SCREENSHOT.is_file() else 0,
        "da3_model_exists": DA3_MODEL.is_file(),
        "da3_model_size": DA3_MODEL.stat().st_size if DA3_MODEL.is_file() else 0,
        "da3_screenshot_exists": DA3_SCREENSHOT.is_file(),
        "da3_screenshot_version": (
            int(DA3_SCREENSHOT.stat().st_mtime_ns) if DA3_SCREENSHOT.is_file() else 0
        ),
    }
    if MANIFEST.is_file():
        try:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
            results.update(
                {
                    "exported_frames": manifest.get("exported_frame_count"),
                    "rejected_frames": manifest.get("rejected_blurry_frames"),
                    "interval": manifest.get("interval_seconds"),
                }
            )
        except (OSError, json.JSONDecodeError):
            pass
    results.update(model_statistics())
    if results.get("points") is None:
        results["points"] = parse_ply_vertex_count(MODEL)
    if DA3_METADATA.is_file():
        try:
            metadata = json.loads(DA3_METADATA.read_text(encoding="utf-8"))
            results.update(
                {
                    "da3_points": metadata.get("point_count"),
                    "da3_images": metadata.get("input_image_count"),
                    "da3_device": metadata.get("device"),
                    "da3_model": metadata.get("model"),
                    "da3_elapsed_seconds": metadata.get("elapsed_seconds"),
                }
            )
        except (OSError, json.JSONDecodeError):
            pass
    if results.get("da3_points") is None:
        results["da3_points"] = parse_ply_vertex_count(DA3_MODEL)
    return results


def da3_python():
    configured = os.environ.get("DA3_PYTHON")
    if configured:
        return configured
    dedicated = REPO_DIR / ".venv-da3" / "bin" / "python"
    return str(dedicated) if dedicated.is_file() else sys.executable


class PipelineService:
    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.cancel_requested = False
        self.state = {
            "status": "idle",
            "stage": "Ready",
            "started_at": None,
            "finished_at": None,
            "log": [],
            "error": None,
            "results": current_results(),
        }

    def videos(self):
        if not VIDEO_DIR.is_dir():
            return []
        return [
            {"name": path.name, "size": path.stat().st_size}
            for path in sorted(VIDEO_DIR.iterdir())
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]

    def snapshot(self):
        with self.lock:
            return {**self.state, "log": list(self.state["log"]), "videos": self.videos()}

    def append_log(self, line):
        with self.lock:
            self.state["log"].append(line.rstrip())
            self.state["log"] = self.state["log"][-1200:]

    def set_stage(self, stage):
        with self.lock:
            self.state["stage"] = stage

    def selected_video(self, name):
        candidates = {item["name"] for item in self.videos()}
        if name not in candidates:
            raise ValueError("Select a video from a5/video/.")
        return VIDEO_DIR / name

    def commands(self, action, payload):
        valid_actions = {
            "full",
            "extract",
            "reconstruct",
            "visualize",
            "da3",
            "visualize_da3",
        }
        if action not in valid_actions:
            raise ValueError(f"Unknown action: {action}")

        interval = bounded_number(payload, "interval", 0.25, 0.05, 10.0)
        max_frames = bounded_number(payload, "max_frames", 100, 3, 500, int)
        min_blur = bounded_number(payload, "min_blur_score", 40.0, 0.0, 10000.0)
        max_size = bounded_number(payload, "max_size", 1600, 320, 4096, int)
        max_features = bounded_number(payload, "max_features", 2048, 256, 16384, int)
        overlap = bounded_number(payload, "sequential_overlap", 5, 1, 50, int)
        da3_max_images = bounded_number(payload, "da3_max_images", 4, 1, 100, int)
        da3_max_points = bounded_number(payload, "da3_max_points", 500000, 1000, 2000000, int)
        da3_confidence = bounded_number(
            payload, "da3_confidence", 20.0, 0.0, 99.0
        )
        da3_resolution = bounded_number(
            payload, "da3_resolution", 392, 0, 1008, int
        )
        if da3_resolution and (da3_resolution < 224 or da3_resolution % 14):
            raise ValueError(
                "da3_resolution must be 0 (automatic) or a multiple of 14 from 224 to 1008"
            )
        da3_device = str(payload.get("da3_device", "auto"))
        if da3_device not in ("auto", "cuda", "mps", "cpu"):
            raise ValueError("da3_device must be auto, cuda, mps, or cpu")
        da3_model = str(payload.get("da3_model", "depth-anything/DA3-SMALL"))
        allowed_da3_models = {
            "depth-anything/DA3-SMALL",
            "depth-anything/DA3-BASE",
            "depth-anything/DA3-LARGE-1.1",
        }
        if da3_model not in allowed_da3_models:
            raise ValueError("Unsupported DA3 model")

        commands = []

        def add_command(stage, command, *, command_id=None, optional=False, depends_on=None):
            commands.append(
                {
                    "stage": stage,
                    "command": command,
                    "id": command_id,
                    "optional": optional,
                    "depends_on": depends_on,
                }
            )

        if action in ("full", "extract"):
            video = self.selected_video(str(payload.get("video", "")))
            add_command(
                "Extracting video frames",
                [
                    sys.executable,
                    str(BASE_DIR / "src" / "extract_video_frames.py"),
                    "--video",
                    str(video),
                    "--interval",
                    str(interval),
                    "--max-frames",
                    str(max_frames),
                    "--min-blur-score",
                    str(min_blur),
                    "--max-size",
                    str(max_size),
                ],
            )
        if action in ("full", "reconstruct"):
            add_command(
                "Running COLMAP reconstruction",
                [
                    sys.executable,
                    str(BASE_DIR / "src" / "run_colmap.py"),
                    "--overwrite",
                    "--max-features",
                    str(max_features),
                    "--sequential-overlap",
                    str(overlap),
                ],
            )
        if action in ("full", "reconstruct", "visualize"):
            add_command(
                "Rendering point cloud",
                [sys.executable, str(BASE_DIR / "src" / "visualize.py")],
            )
        if action in ("full", "da3"):
            add_command(
                "Running Depth Anything 3 reconstruction",
                [
                    da3_python(),
                    str(BASE_DIR / "src" / "run_da3.py"),
                    "--images",
                    str(IMAGE_DIR),
                    "--output-dir",
                    str(BASE_DIR / "da3"),
                    "--device",
                    da3_device,
                    "--model",
                    da3_model,
                    "--max-images",
                    str(da3_max_images),
                    "--max-points",
                    str(da3_max_points),
                    "--confidence-percentile",
                    str(da3_confidence),
                    "--resolution",
                    str(da3_resolution),
                ],
                command_id="da3",
                optional=action == "full",
            )
        if action in ("full", "da3", "visualize_da3"):
            if action == "visualize_da3" and not DA3_MODEL.is_file():
                raise RuntimeError(
                    "No DA3 point cloud exists yet. Run 'DA3 + image' successfully first."
                )
            add_command(
                "Rendering Depth Anything 3 point cloud",
                [
                    sys.executable,
                    str(BASE_DIR / "src" / "visualize.py"),
                    str(DA3_MODEL),
                    "--screenshot",
                    str(DA3_SCREENSHOT),
                    "--title",
                    "A5 - Depth Anything 3 reconstruction",
                    "--point-radius",
                    "2",
                ],
                optional=action == "full",
                depends_on="da3" if action in ("full", "da3") else None,
            )
        return commands

    def start(self, action, payload):
        commands = self.commands(action, payload)
        with self.lock:
            if self.state["status"] == "running":
                raise RuntimeError("A pipeline job is already running.")
            self.cancel_requested = False
            self.state = {
                "status": "running",
                "stage": "Starting",
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "finished_at": None,
                "log": [],
                "error": None,
                "results": self.state.get("results", {}),
            }
        threading.Thread(target=self.run_job, args=(commands,), daemon=True).start()
        return self.snapshot()

    def run_job(self, commands):
        failed_commands = set()
        warnings = []
        try:
            for item in commands:
                stage = item["stage"]
                command = item["command"]
                dependency = item.get("depends_on")
                if dependency in failed_commands:
                    warning = f"Skipping {stage}: the required DA3 stage did not finish."
                    warnings.append(warning)
                    self.append_log(f"WARNING: {warning}")
                    continue
                if self.cancel_requested:
                    raise InterruptedError("Pipeline cancelled.")
                self.set_stage(stage)
                self.append_log("$ " + " ".join(command))
                process = subprocess.Popen(
                    command,
                    cwd=REPO_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                with self.lock:
                    self.process = process
                for line in process.stdout:
                    self.append_log(line)
                return_code = process.wait()
                with self.lock:
                    self.process = None
                if self.cancel_requested:
                    raise InterruptedError("Pipeline cancelled.")
                if return_code != 0:
                    error = f"{stage} failed with exit code {return_code}."
                    if return_code == -signal.SIGKILL and item.get("id") == "da3":
                        error += (
                            " macOS killed DA3 because memory was exhausted. Close memory-heavy "
                            "applications and retry the native MPS low-memory profile."
                        )
                    if item.get("optional"):
                        if item.get("id"):
                            failed_commands.add(item["id"])
                        warnings.append(error)
                        self.append_log(
                            f"WARNING: {error} The successful COLMAP result is retained."
                        )
                        continue
                    raise RuntimeError(error)

            results = current_results()
            final_stage = "Finished with warnings" if warnings else "Finished"
            with self.lock:
                self.state.update(
                    {
                        "status": "success",
                        "stage": final_stage,
                        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "error": " ".join(warnings) if warnings else None,
                        "results": results,
                    }
                )
        except InterruptedError as error:
            with self.lock:
                self.state.update(
                    {
                        "status": "cancelled",
                        "stage": "Cancelled",
                        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "error": str(error),
                    }
                )
        except Exception as error:
            self.append_log(f"ERROR: {error}")
            with self.lock:
                self.state.update(
                    {
                        "status": "error",
                        "stage": "Failed",
                        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "error": str(error),
                    }
                )
        finally:
            with self.lock:
                self.process = None

    def cancel(self):
        with self.lock:
            if self.state["status"] != "running":
                process = None
            else:
                self.cancel_requested = True
                process = self.process
                self.state["stage"] = "Stopping"
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        return self.snapshot()


class Handler(BaseHTTPRequestHandler):
    service = PipelineService()

    def log_message(self, format, *args):
        return

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, download_name=None):
        try:
            body = path.resolve().read_bytes()
        except OSError:
            self.send_error(404)
            return
        mime_type, _ = mimetypes.guess_type(str(path))
        self.send_response(200)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            self.send_json(self.service.snapshot())
        elif path == "/result/screenshot":
            self.send_file(SCREENSHOT)
        elif path == "/result/model":
            self.send_file(MODEL, MODEL.name)
        elif path == "/result/manifest":
            self.send_file(MANIFEST, MANIFEST.name)
        elif path == "/result/da3-screenshot":
            self.send_file(DA3_SCREENSHOT)
        elif path == "/result/da3-model":
            self.send_file(DA3_MODEL, "da3_points.ply")
        elif path == "/result/da3-metadata":
            self.send_file(DA3_METADATA, DA3_METADATA.name)
        elif path == "/":
            self.send_file(WEB_DIR / "index.html")
        elif path.startswith("/static/"):
            relative = Path(path[len("/static/"):])
            target = (STATIC_DIR / relative).resolve()
            static_root = STATIC_DIR.resolve()
            if target != static_root and static_root in target.parents:
                self.send_file(target)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            path = urlparse(self.path).path
            if path == "/api/run":
                self.send_json(self.service.start(str(payload.get("action", "full")), payload))
            elif path == "/api/cancel":
                self.send_json(self.service.cancel())
            else:
                self.send_error(404)
        except (ValueError, RuntimeError) as error:
            self.send_json({"error": str(error)}, status=400)
        except Exception as error:
            self.send_json({"error": str(error)}, status=500)


def find_available_port(host, start_port, max_tries):
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found in range {start_port}-{start_port + max_tries - 1}")


def parse_args():
    parser = argparse.ArgumentParser(description="Browser UI for the A5 reconstruction pipeline.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--max-port-tries", type=int, default=20)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    port = find_available_port(args.host, args.port, args.max_port_tries)
    server = ThreadingHTTPServer((args.host, port), Handler)
    browser_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{browser_host}:{port}"
    print(f"A5 web UI running at {url}")
    if port != args.port:
        print(f"Requested port {args.port} was busy, using {port} instead.")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
