"""Frame extractor for the Helmet Violation Detection project.

Extracts frames from a video file at a configurable interval, computes
per-frame stats, and saves a manifest CSV.  Near-duplicate frames (based on
consecutive brightness and blur similarity) are skipped.
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src import PROJECT_ROOT, RESULTS_DIR


_BRIGHTNESS_DUP_THRESHOLD = 1.5   # max mean brightness diff for near-duplicate
_BLUR_DUP_THRESHOLD = 5.0          # max blur score diff for near-duplicate


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    interval_frames: int = 30,
    interval_seconds: float = None,
    max_frames: int = None,
    skip_near_duplicates: bool = True,
) -> list[dict]:
    """Extract frames from video.

    Parameters
    ----------
    video_path:
        Path to the video file.
    output_dir:
        Directory where JPEG frames are saved.
    interval_frames:
        Extract every N frames. Overridden if interval_seconds is set.
    interval_seconds:
        Extract every T seconds (converted to frames using video FPS).
    max_frames:
        Maximum number of frames to extract (None = no limit).
    skip_near_duplicates:
        Skip frames that are very similar to the previous extracted frame.

    Returns
    -------
    list of dict
        One entry per extracted frame, matching the manifest CSV columns.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resolve interval
    if interval_seconds is not None and fps > 0:
        interval_frames = max(1, int(round(fps * interval_seconds)))
    interval_frames = max(1, interval_frames)

    manifest = []
    frame_index = 0
    extracted_count = 0
    prev_brightness = None
    prev_blur = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_index % interval_frames == 0:
                timestamp_s = frame_index / fps if fps > 0 else 0.0
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = float(np.mean(gray))
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                # Near-duplicate check
                is_near_dup = False
                if skip_near_duplicates and prev_brightness is not None:
                    b_diff = abs(mean_brightness - prev_brightness)
                    v_diff = abs(blur_score - prev_blur)
                    if b_diff < _BRIGHTNESS_DUP_THRESHOLD and v_diff < _BLUR_DUP_THRESHOLD:
                        is_near_dup = True

                if not is_near_dup:
                    file_name = f"frame_{frame_index:08d}.jpg"
                    file_path = output_dir / file_name
                    cv2.imwrite(
                        str(file_path), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95]
                    )
                    entry = {
                        "frame_index": frame_index,
                        "timestamp_s": round(timestamp_s, 4),
                        "file_path": str(file_path),
                        "width": width,
                        "height": height,
                        "mean_brightness": round(mean_brightness, 3),
                        "blur_score": round(blur_score, 3),
                    }
                    manifest.append(entry)
                    prev_brightness = mean_brightness
                    prev_blur = blur_score
                    extracted_count += 1
                    if max_frames is not None and extracted_count >= max_frames:
                        break

            frame_index += 1
    finally:
        cap.release()

    return manifest


def save_manifest_csv(manifest: list[dict], output_path: Path = None) -> Path:
    """Save the frame manifest as a CSV file."""
    out = output_path or RESULTS_DIR / "frame_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index", "timestamp_s", "file_path",
        "width", "height", "mean_brightness", "blur_score",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    return out


def main() -> int:
    """Entry point."""
    import yaml

    config_path = PROJECT_ROOT / "configs" / "base_config.yaml"
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    video_path = Path(cfg.get("video_path", ""))
    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 1

    output_dir = PROJECT_ROOT / "data" / "frames"
    print(f"Extracting frames from: {video_path}")
    print(f"Output directory: {output_dir}")
    t0 = time.perf_counter()
    try:
        manifest = extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            interval_frames=30,
            skip_near_duplicates=True,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - t0

    manifest_path = save_manifest_csv(manifest)
    print(f"Extracted {len(manifest)} frames in {elapsed:.2f}s.")
    print(f"Manifest saved to: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
