import argparse
import base64
import json
import mimetypes
import socket
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from simple_stereo import (
    CALIBRATION,
    GROUND_TRUTH,
    LEFT_IMAGE,
    OUTPUT_DIR,
    RIGHT_IMAGE,
    compute_depth,
    compute_disparity,
    error_map_for_display,
    evaluate_disparity,
    load_calibration,
    load_stereo_images,
    make_comparison_image,
    normalize_for_display,
    read_pfm,
    save_outputs,
    valid_disparity_mask,
)

TESTS_PATH = OUTPUT_DIR / "tests.json"
WEB_DIR = Path(__file__).with_name("web")
STATIC_DIR = WEB_DIR / "static"


def image_to_base64(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def resize_large(image, max_width=1400):
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / width
    return cv2.resize(image, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)


def parse_params(payload):
    algorithm = payload.get("algorithm", "sgbm")
    if algorithm not in ("sgbm", "bm"):
        raise ValueError("algorithm must be 'sgbm' or 'bm'")

    block_size = int(payload.get("block_size", 5))
    min_block_size = 5 if algorithm == "bm" else 3
    if block_size < min_block_size:
        block_size = min_block_size
    if block_size % 2 == 0:
        block_size += 1

    num_disparities = int(payload.get("num_disparities", 176))
    num_disparities = max(16, min(320, int(np.ceil(num_disparities / 16.0) * 16)))

    return {
        "algorithm": algorithm,
        "block_size": block_size,
        "uniqueness_ratio": int(payload.get("uniqueness_ratio", 5)),
        "speckle_window_size": int(payload.get("speckle_window_size", 100)),
        "speckle_range": int(payload.get("speckle_range", 2)),
        "min_disparity": int(payload.get("min_disparity", 0)),
        "num_disparities": num_disparities,
    }


def public_params(params):
    return dict(params)


class StereoService:
    def __init__(self):
        self.calib = load_calibration(CALIBRATION)
        self.left, self.right = load_stereo_images(LEFT_IMAGE, RIGHT_IMAGE)
        self.ground_truth, _ = read_pfm(GROUND_TRUTH)
        self.last_result = None
        self.last_params = None
        self.test_lock = threading.Lock()
        self.tests = self.load_tests()

    def next_test_id(self):
        ids = [item.get("id") for item in self.tests if isinstance(item.get("id"), int)]
        return max(ids, default=0) + 1

    def compute(self, params):
        disparity = compute_disparity(
            self.left,
            self.right,
            self.calib,
            params["algorithm"],
            params["block_size"],
            params["uniqueness_ratio"],
            params["speckle_window_size"],
            params["speckle_range"],
            params["min_disparity"],
            params["num_disparities"],
        )
        depth = compute_depth(disparity, self.calib)
        metrics = evaluate_disparity(self.ground_truth, disparity, self.calib)
        self.last_result = {"disparity": disparity, "depth": depth, "metrics": metrics}
        self.last_params = params
        return self.response(params, self.last_result, "Computed current setting.")

    def response(self, params, result, status):
        disparity = result["disparity"]
        depth = result["depth"]
        disp_vis = normalize_for_display(disparity, valid_disparity_mask(disparity))
        gt_vis = normalize_for_display(self.ground_truth, valid_disparity_mask(self.ground_truth, self.calib))
        error_vis = error_map_for_display(self.ground_truth, disparity, self.calib)
        depth_vis = normalize_for_display(depth, np.isfinite(depth), cv2.COLORMAP_VIRIDIS)
        comparison = make_comparison_image(
            self.left,
            disparity,
            self.ground_truth,
            depth,
            self.calib,
            f"{params['algorithm']} b{params['block_size']}",
        )

        return {
            "params": params,
            "metrics": result["metrics"],
            "status": status,
            "images": {
                "comparison": image_to_base64(resize_large(comparison)),
                "left": image_to_base64(resize_large(self.left)),
                "disparity": image_to_base64(resize_large(disp_vis)),
                "ground_truth": image_to_base64(resize_large(gt_vis)),
                "error": image_to_base64(resize_large(error_vis)),
                "depth": image_to_base64(resize_large(depth_vis)),
            },
        }

    def save(self, params):
        if self.last_result is None or self.last_params != params:
            self.compute(params)
        label = (
            f"{params['algorithm']}_b{params['block_size']}"
            f"_u{params['uniqueness_ratio']}"
            f"_md{params['min_disparity']}"
            f"_nd{params['num_disparities']}"
            f"_sw{params['speckle_window_size']}"
            f"_sr{params['speckle_range']}"
        )
        save_outputs(
            OUTPUT_DIR,
            self.left,
            self.last_result["disparity"],
            self.ground_truth,
            self.last_result["depth"],
            self.calib,
            label,
        )
        return {"status": f"Exported current result to {OUTPUT_DIR} with label {label}."}

    def load_tests(self):
        if not TESTS_PATH.exists():
            return []
        try:
            with TESTS_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, list):
                return data
        except (OSError, json.JSONDecodeError):
            pass
        return []

    def write_tests(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with TESTS_PATH.open("w", encoding="utf-8") as file:
            json.dump(self.tests, file, indent=2)

    def list_tests(self):
        with self.test_lock:
            return {"tests": list(self.tests)}

    def save_test(self, params):
        if self.last_result is None or self.last_params != params:
            self.compute(params)

        existing = next((item for item in self.tests if item.get("params") == params), None)
        entry = {
            "id": existing.get("id") if existing else self.next_test_id(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": public_params(params),
            "metrics": self.last_result["metrics"],
        }

        with self.test_lock:
            self.tests = [
                item for item in self.tests
                if item.get("params") != entry["params"]
            ]
            self.tests.append(entry)
            self.write_tests()
            tests = list(self.tests)

        return {
            "status": f"Saved test #{entry['id']} to {TESTS_PATH}.",
            "tests": tests,
        }


class Handler(BaseHTTPRequestHandler):
    service = StereoService()

    def log_message(self, format, *args):
        return

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        resolved = path.resolve()
        mime_type, _ = mimetypes.guess_type(str(resolved))
        try:
            body = resolved.read_bytes()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/tests":
            self.send_json(self.service.list_tests())
            return

        if path == "/":
            self.send_file(WEB_DIR / "index.html")
            return

        if path.startswith("/static/"):
            relative = Path(path[len("/static/"):])
            target = (STATIC_DIR / relative).resolve()
            static_root = STATIC_DIR.resolve()
            if static_root not in target.parents and target != static_root:
                self.send_error(404)
                return
            self.send_file(target)
            return

        if path == "/favicon.ico":
            self.send_error(404)
            return

        if path != "/":
            self.send_error(404)
            return

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            path = urlparse(self.path).path

            if path == "/api/compute":
                self.send_json(self.service.compute(parse_params(payload)))
            elif path == "/api/save":
                self.send_json(self.service.save(parse_params(payload)))
            elif path == "/api/test":
                self.send_json(self.service.save_test(parse_params(payload)))
            else:
                self.send_error(404)
        except Exception as error:
            self.send_json({"error": str(error)}, status=500)


def parse_args():
    parser = argparse.ArgumentParser(description="Browser UI for A3 Simple Stereo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-port-tries", type=int, default=20)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


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


def main():
    args = parse_args()
    port = find_available_port(args.host, args.port, args.max_port_tries)
    server = ThreadingHTTPServer((args.host, port), Handler)
    url = f"http://{args.host}:{port}"
    print(f"A3 web UI running at {url}")
    if port != args.port:
        print(f"Requested port {args.port} was busy, using {port} instead.")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
