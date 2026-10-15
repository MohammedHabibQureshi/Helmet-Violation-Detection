"""Video inspector for the Helmet Violation Detection project.

Analyses video properties, frame decodability, brightness, and motion blur.
Saves results to results/video_inspection.json.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src import PROJECT_ROOT, RESULTS_DIR


def inspect_video(video_path: str | Path) -> dict:
    """Inspect a video file and return a dict of properties.

    Parameters
    ----------
    video_path:
        Path to the video file.

    Returns
    -------
    dict
        Full inspection report. Raises FileNotFoundError if the file does not exist.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_fourcc = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))
        file_size_mb = round(video_path.stat().st_size / (1024 ** 2), 2)

        duration_s = frame_count / fps if fps > 0 else 0.0

        # Sample frames at key positions
        sample_fractions = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
        sample_indices = [
            min(int(f * (frame_count - 1)), frame_count - 1)
            for f in sample_fractions
        ]
        # Deduplicate while preserving order
        seen = set()
        sample_indices_unique = []
        for idx in sample_indices:
            if idx not in seen:
                seen.add(idx)
                sample_indices_unique.append(idx)

        sample_results = []
        brightness_values = []
        blur_values = []

        for frame_idx in sample_indices_unique:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
            ret, frame = cap.read()
            decode_success = ret and frame is not None

            mean_brightness = None
            std_brightness = None
            blur_score = None
            timestamp_s = frame_idx / fps if fps > 0 else 0.0

            if decode_success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = float(np.mean(gray))
                std_brightness = float(np.std(gray))
                # Laplacian variance as motion blur indicator (lower = blurrier)
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness_values.append(mean_brightness)
                blur_values.append(blur_score)

            sample_results.append({
                "frame_index": frame_idx,
                "position_fraction": frame_idx / max(frame_count - 1, 1),
                "timestamp_s": round(timestamp_s, 4),
                "decode_success": decode_success,
                "mean_brightness": round(mean_brightness, 3) if mean_brightness is not None else None,
                "std_brightness": round(std_brightness, 3) if std_brightness is not None else None,
                "blur_score": round(blur_score, 3) if blur_score is not None else None,
            })

        # Aggregate stats
        decodable_samples = sum(1 for s in sample_results if s["decode_success"])
        decodable_fraction = decodable_samples / len(sample_results)
        estimated_decodable_frames = int(frame_count * decodable_fraction)

        overall_mean_brightness = (
            round(float(np.mean(brightness_values)), 3)
            if brightness_values else None
        )
        overall_std_brightness = (
            round(float(np.std(brightness_values)), 3)
            if brightness_values else None
        )
        overall_mean_blur = (
            round(float(np.mean(blur_values)), 3) if blur_values else None
        )
        blur_min = round(float(np.min(blur_values)), 3) if blur_values else None
        blur_max = round(float(np.max(blur_values)), 3) if blur_values else None

        # Estimate actual frame count by seeking to known positions
        actual_frame_count_estimate = estimated_decodable_frames

    finally:
        cap.release()

    report = {
        "video_path": str(video_path),
        "file_size_mb": file_size_mb,
        "width": width,
        "height": height,
        "fps": round(fps, 4),
        "frame_count": frame_count,
        "duration_s": round(duration_s, 4),
        "codec_fourcc": codec_fourcc.strip("\x00"),
        "sample_frames": sample_results,
        "decodable_samples": decodable_samples,
        "total_samples": len(sample_results),
        "decodable_fraction": round(decodable_fraction, 4),
        "estimated_decodable_frames": estimated_decodable_frames,
        "overall_mean_brightness": overall_mean_brightness,
        "overall_std_brightness": overall_std_brightness,
        "overall_mean_blur_score": overall_mean_blur,
        "blur_score_min": blur_min,
        "blur_score_max": blur_max,
        "inspection_timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
    }
    return report


def save_inspection_json(report: dict, output_path: Path = None) -> Path:
    """Save inspection report to JSON."""
    out = output_path or RESULTS_DIR / "video_inspection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
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

    print(f"Inspecting video: {video_path}")
    t0 = time.perf_counter()
    try:
        report = inspect_video(video_path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - t0

    out_path = save_inspection_json(report)
    print(f"Inspection complete in {elapsed:.2f}s. Report: {out_path}")
    print(f"  Resolution:   {report['width']}x{report['height']}")
    print(f"  FPS:          {report['fps']}")
    print(f"  Frame count:  {report['frame_count']}")
    print(f"  Duration:     {report['duration_s']:.2f}s")
    print(f"  Codec:        {report['codec_fourcc']}")
    print(f"  Brightness:   mean={report['overall_mean_brightness']}, "
          f"std={report['overall_std_brightness']}")
    print(f"  Blur score:   mean={report['overall_mean_blur_score']} "
          f"(min={report['blur_score_min']}, max={report['blur_score_max']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
