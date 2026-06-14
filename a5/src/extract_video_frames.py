import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = BASE_DIR / "video"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "scene" / "images"
DEFAULT_MANIFEST = BASE_DIR / "data" / "scene" / "video_frames.json"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def find_video(video_dir):
    videos = sorted(
        path for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(
            f"No video found in {video_dir}. Add an MP4, MOV, M4V, AVI, MKV, or WEBM file."
        )
    if len(videos) > 1:
        names = ", ".join(path.name for path in videos)
        raise RuntimeError(f"Multiple videos found ({names}). Select one with --video.")
    return videos[0]


def resize_frame(frame, max_size):
    if max_size <= 0:
        return frame
    height, width = frame.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_size:
        return frame
    scale = max_size / float(longest_side)
    size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def parse_frame_rate(value):
    numerator, denominator = value.split("/", maxsplit=1)
    denominator = float(denominator)
    return float(numerator) / denominator if denominator else 0.0


def probe_video(video_path, timeout):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise RuntimeError(f"No video stream found in: {video_path}")

    stream = data["streams"][0]
    fps = parse_frame_rate(stream.get("avg_frame_rate", "0/1"))
    duration_value = stream.get("duration") or data.get("format", {}).get("duration")
    duration = float(duration_value) if duration_value is not None else None
    frame_count_value = stream.get("nb_frames")
    source_frames = int(frame_count_value) if frame_count_value not in (None, "N/A") else 0
    if source_frames <= 0 and fps > 0 and duration is not None:
        source_frames = int(round(fps * duration))
    return fps, source_frames, duration


def save_frame(frame, timestamp, args, exported):
    score = blur_score(frame)
    if score < args.min_blur_score:
        return False

    filename = f"frame_{len(exported):04d}_{timestamp:08.2f}s.jpg"
    output_path = args.output_dir / filename
    written = cv2.imwrite(
        str(output_path),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
    )
    if not written:
        raise RuntimeError(f"Could not write frame: {output_path}")
    exported.append(
        {
            "file": filename,
            "time_seconds": round(timestamp, 3),
            "blur_score": round(score, 2),
            "width": frame.shape[1],
            "height": frame.shape[0],
        }
    )
    return True


def extract_with_ffmpeg(video_path, args):
    filters = [f"fps={1.0 / args.interval:.12g}"]
    if args.max_size > 0:
        filters.append(
            f"scale={args.max_size}:{args.max_size}:force_original_aspect_ratio=decrease"
        )

    with tempfile.TemporaryDirectory(prefix="a5-video-frames-") as temp_dir:
        temp_path = Path(temp_dir)
        local_video = temp_path / f"source{video_path.suffix.lower()}"
        shutil.copyfile(video_path, local_video)
        fps, source_frames, duration = probe_video(local_video, args.decode_timeout)
        candidate_pattern = str(temp_path / "candidate_%04d.jpg")
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "2",
            "-ss",
            str(args.start),
            "-i",
            str(local_video),
        ]
        if args.end is not None:
            command.extend(["-t", str(args.end - args.start)])
        command.extend(
            [
                "-vf",
                ",".join(filters),
                "-q:v",
                "2",
                candidate_pattern,
            ]
        )
        subprocess.run(command, check=True, timeout=args.decode_timeout)

        exported = []
        rejected_blurry = 0
        for index, candidate in enumerate(sorted(Path(temp_dir).glob("candidate_*.jpg"))):
            if len(exported) >= args.max_frames:
                break
            frame = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Could not read extracted frame: {candidate}")
            timestamp = args.start + index * args.interval
            if not save_frame(frame, timestamp, args, exported):
                rejected_blurry += 1

    return fps, source_frames, duration, exported, rejected_blurry, "ffmpeg"


def extract_with_opencv(video_path, args):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        capture.release()
        raise RuntimeError("Video does not report a valid frame rate")
    duration = source_frames / fps if source_frames > 0 else None
    end_time = args.end if args.end is not None else duration
    start_frame = int(round(args.start * fps))
    target_frame = float(start_frame)
    current_frame = start_frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    exported = []
    rejected_blurry = 0

    while len(exported) < args.max_frames:
        timestamp = target_frame / fps
        if end_time is not None and timestamp > end_time:
            break

        target_index = int(round(target_frame))
        while current_frame < target_index:
            if not capture.grab():
                capture.release()
                return fps, source_frames, duration, exported, rejected_blurry, "opencv"
            current_frame += 1

        ok, frame = capture.read()
        if not ok:
            break
        current_frame += 1
        frame = resize_frame(frame, args.max_size)
        if not save_frame(frame, timestamp, args, exported):
            rejected_blurry += 1
        target_frame += args.interval * fps

    capture.release()
    return fps, source_frames, duration, exported, rejected_blurry, "opencv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract evenly spaced reconstruction images from a video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Video file. If omitted, use the only video in a5/video/.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between frames.")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--end", type=float, default=None, help="Optional end time in seconds.")
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--max-size", type=int, default=1600)
    parser.add_argument("--min-blur-score", type=float, default=40.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--decode-timeout",
        type=float,
        default=180.0,
        help="Maximum seconds allowed for ffprobe or ffmpeg.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    if args.start < 0:
        raise ValueError("--start cannot be negative")
    if args.end is not None and args.end <= args.start:
        raise ValueError("--end must be greater than --start")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if args.decode_timeout <= 0:
        raise ValueError("--decode-timeout must be positive")

    video_path = args.video or find_video(DEFAULT_VIDEO_DIR)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in args.output_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        fps, source_frames, duration, exported, rejected_blurry, backend = extract_with_ffmpeg(
            video_path, args
        )
    else:
        fps, source_frames, duration, exported, rejected_blurry, backend = extract_with_opencv(
            video_path, args
        )
    if len(exported) < 3:
        raise RuntimeError(
            f"Only {len(exported)} usable frames were extracted. "
            "Use a longer video, a smaller --interval, or a lower --min-blur-score."
        )

    manifest = {
        "source_video": str(video_path),
        "source_fps": fps,
        "source_frame_count": source_frames,
        "source_duration_seconds": duration,
        "decoder_backend": backend,
        "interval_seconds": args.interval,
        "minimum_blur_score": args.min_blur_score,
        "rejected_blurry_frames": rejected_blurry,
        "exported_frame_count": len(exported),
        "frames": exported,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Video: {video_path}")
    print(f"Extracted: {len(exported)} frames to {args.output_dir}")
    print(f"Rejected as blurry: {rejected_blurry}")
    print(f"Decoder: {backend}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
