"""Sequential single-threaded video processing pipeline.

Processes video frame-by-frame: detect -> associate -> crop -> preprocess
-> OCR -> suppress duplicates. Records per-frame timing and all counters.
"""

import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src import PROJECT_ROOT, RESULTS_DIR
from src.detector import YOLODetector, DetectorConfig
from src.helmet_association import associate_violations, AssociationConfig
from src.plate_preprocessor import PlatePreprocessor
from src.ocr_engines import get_available_engines, NullOCREngine, OCRResult
from src.duplicate_suppressor import DuplicateSuppressor


@dataclass
class PipelineConfig:
    """Configuration for the sequential pipeline."""

    device: str = "auto"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    frame_skip: int = 1           # Process every N-th frame (1 = all)
    max_frames: Optional[int] = None
    queue_capacity: int = 64      # Unused in sequential mode, kept for parity
    ocr_engine: str = "easyocr"   # Name of OCR engine to use
    preprocessing_pipeline: str = "D"
    duplicate_window_s: float = 5.0
    warmup_frames: int = 3


@dataclass
class FrameRecord:
    """Per-frame timing and result record."""

    frame_index: int
    timestamp_s: float
    detect_ms: float
    associate_ms: float
    preprocess_ms: float
    ocr_ms: float
    total_ms: float
    detections_count: int
    violations_found: int
    crops_generated: int
    events_completed: int


class SequentialPipeline:
    """Single-threaded video processing pipeline.

    Parameters
    ----------
    config:
        Pipeline configuration.
    ocr_engine:
        Pre-instantiated OCR engine. If None, resolves from config.
    """

    def __init__(
        self,
        config: PipelineConfig = None,
        ocr_engine=None,
    ):
        self.config = config or PipelineConfig()
        cfg = self.config

        # Detector
        self._detector = YOLODetector(
            DetectorConfig(
                device=cfg.device,
                conf=cfg.conf_threshold,
                iou=cfg.iou_threshold,
                warmup_runs=cfg.warmup_frames,
            )
        )

        # OCR engine
        if ocr_engine is not None:
            self._ocr = ocr_engine
        else:
            available = get_available_engines()
            self._ocr = available.get(cfg.ocr_engine) or NullOCREngine()

        self._preprocessor = PlatePreprocessor()
        self._suppressor = DuplicateSuppressor(
            window_seconds=cfg.duplicate_window_s
        )
        self._assoc_config = AssociationConfig()

        # Counters
        self._counters = {
            "frames_read": 0,
            "frames_processed": 0,
            "frames_skipped": 0,
            "crops_generated": 0,
            "crops_processed": 0,
            "crops_rejected": 0,
            "duplicate_events_suppressed": 0,
            "completed_events": 0,
            "association_candidates": 0,
            "association_accepted": 0,
        }
        self._frame_records: list[FrameRecord] = []
        self._events: list[dict] = []

    def _warmup(self) -> None:
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(self.config.warmup_frames):
            self._detector.detect_frame(blank)

    def _process_frame(
        self, frame: np.ndarray, frame_index: int, timestamp_s: float
    ) -> FrameRecord:
        """Process a single frame end-to-end."""
        t_total_start = time.perf_counter()

        # Detect
        t0 = time.perf_counter()
        detections = self._detector.detect_frame(
            frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
        )
        detect_ms = (time.perf_counter() - t0) * 1000

        # Associate
        t0 = time.perf_counter()
        candidates = associate_violations(
            detections, frame_index, timestamp_s, self._assoc_config
        )
        associate_ms = (time.perf_counter() - t0) * 1000
        self._counters["association_candidates"] += len(candidates)
        accepted = [c for c in candidates if c["accepted"]]
        self._counters["association_accepted"] += len(accepted)

        preprocess_ms = 0.0
        ocr_ms = 0.0
        crops_this_frame = 0
        events_this_frame = 0

        for candidate in accepted:
            plate_box = candidate.get("plate_box")
            if plate_box is None:
                self._counters["crops_rejected"] += 1
                continue

            # Crop + preprocess
            t0 = time.perf_counter()
            crop = self._preprocessor.extract_plate_crop(frame, plate_box)
            if crop is None:
                preprocess_ms += (time.perf_counter() - t0) * 1000
                self._counters["crops_rejected"] += 1
                continue
            pipeline_result = self._preprocessor.run_pipeline(
                crop, self.config.preprocessing_pipeline
            )
            preprocess_ms += (time.perf_counter() - t0) * 1000
            crops_this_frame += 1
            self._counters["crops_generated"] += 1

            processed_image = pipeline_result.get("image")
            if processed_image is None:
                self._counters["crops_rejected"] += 1
                continue

            # OCR
            t0 = time.perf_counter()
            ocr_result: OCRResult = self._ocr.recognize(processed_image)
            ocr_ms += (time.perf_counter() - t0) * 1000
            self._counters["crops_processed"] += 1

            plate_text = ocr_result.text_corrected or ocr_result.text_normalized
            if not plate_text:
                self._counters["crops_rejected"] += 1
                continue

            # Duplicate suppression
            if self._suppressor.is_duplicate(plate_text, timestamp_s):
                self._counters["duplicate_events_suppressed"] += 1
                continue

            # Completed event
            self._counters["completed_events"] += 1
            events_this_frame += 1
            self._events.append({
                "frame_index": frame_index,
                "timestamp_s": round(timestamp_s, 4),
                "plate_text_raw": ocr_result.text_raw,
                "plate_text_normalized": ocr_result.text_normalized,
                "plate_text_corrected": plate_text,
                "ocr_confidence": ocr_result.confidence,
                "ocr_engine": ocr_result.engine_name,
                "preprocessing_pipeline": self.config.preprocessing_pipeline,
                "plate_box": plate_box,
                "detect_ms": round(detect_ms, 3),
                "preprocess_ms": round(preprocess_ms, 3),
                "ocr_ms": round(ocr_ms, 3),
            })

        total_ms = (time.perf_counter() - t_total_start) * 1000
        return FrameRecord(
            frame_index=frame_index,
            timestamp_s=round(timestamp_s, 4),
            detect_ms=round(detect_ms, 3),
            associate_ms=round(associate_ms, 3),
            preprocess_ms=round(preprocess_ms, 3),
            ocr_ms=round(ocr_ms, 3),
            total_ms=round(total_ms, 3),
            detections_count=len(detections),
            violations_found=len(accepted),
            crops_generated=crops_this_frame,
            events_completed=events_this_frame,
        )

    def run(
        self,
        video_path: str | Path,
        output_dir: str | Path = None,
    ) -> dict:
        """Run the pipeline on a video.

        Parameters
        ----------
        video_path:
            Path to the video file.
        output_dir:
            Directory for results. Defaults to ``results/``.

        Returns
        -------
        dict
            Summary of the run.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir) if output_dir else RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[SequentialPipeline] Warmup ({self.config.warmup_frames} frames)...")
        self._warmup()

        print(f"[SequentialPipeline] Processing video: {video_path.name}")
        wall_start = time.perf_counter()

        try:
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                self._counters["frames_read"] += 1

                if frame_index % self.config.frame_skip != 0:
                    self._counters["frames_skipped"] += 1
                    frame_index += 1
                    continue

                timestamp_s = frame_index / fps if fps > 0 else 0.0
                record = self._process_frame(frame, frame_index, timestamp_s)
                self._frame_records.append(record)
                self._counters["frames_processed"] += 1

                if (
                    self.config.max_frames is not None
                    and self._counters["frames_processed"] >= self.config.max_frames
                ):
                    break
                frame_index += 1
        finally:
            cap.release()

        wall_s = time.perf_counter() - wall_start
        processed = self._counters["frames_processed"]
        achieved_fps = processed / wall_s if wall_s > 0 else 0.0

        # Latencies
        latencies = [r.total_ms for r in self._frame_records]
        import statistics as _stats
        mean_latency = _stats.mean(latencies) if latencies else 0.0
        median_latency = _stats.median(latencies) if latencies else 0.0
        p90 = sorted(latencies)[int(0.90 * len(latencies))] if latencies else 0.0
        p95 = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0.0
        max_latency = max(latencies) if latencies else 0.0

        summary = {
            "mode": "sequential",
            "device": self.config.device,
            "ocr_engine": self.config.ocr_engine,
            "preprocessing_pipeline": self.config.preprocessing_pipeline,
            "total_wall_s": round(wall_s, 4),
            "achieved_fps": round(achieved_fps, 4),
            "mean_latency_ms": round(mean_latency, 3),
            "median_latency_ms": round(median_latency, 3),
            "p90_latency_ms": round(p90, 3),
            "p95_latency_ms": round(p95, 3),
            "max_latency_ms": round(max_latency, 3),
            **self._counters,
            "suppressor_stats": self._suppressor.get_stats(),
        }

        # Save results CSV
        csv_path = output_dir / "sequential_results.csv"
        _save_frame_records_csv(self._frame_records, csv_path)

        # Save events CSV
        events_csv = output_dir / "sequential_events.csv"
        _save_events_csv(self._events, events_csv)

        # Save summary JSON
        summary_path = output_dir / "sequential_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)

        print(f"[SequentialPipeline] Done: {processed} frames in {wall_s:.2f}s "
              f"({achieved_fps:.2f} fps)")
        print(f"  Completed events: {self._counters['completed_events']}")
        print(f"  Results: {csv_path}")
        print(f"  Summary: {summary_path}")
        return summary


def _save_frame_records_csv(records: list[FrameRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_index", "timestamp_s", "detect_ms", "associate_ms",
        "preprocess_ms", "ocr_ms", "total_ms", "detections_count",
        "violations_found", "crops_generated", "events_completed",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({
                "frame_index": r.frame_index,
                "timestamp_s": r.timestamp_s,
                "detect_ms": r.detect_ms,
                "associate_ms": r.associate_ms,
                "preprocess_ms": r.preprocess_ms,
                "ocr_ms": r.ocr_ms,
                "total_ms": r.total_ms,
                "detections_count": r.detections_count,
                "violations_found": r.violations_found,
                "crops_generated": r.crops_generated,
                "events_completed": r.events_completed,
            })


def _save_events_csv(events: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not events:
        path.write_text("", encoding="utf-8")
        return
    fields = list(events[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(events)


def run(
    video_path: str | Path,
    output_dir: str | Path = None,
    config: PipelineConfig = None,
) -> dict:
    """Module-level convenience function."""
    pipeline = SequentialPipeline(config=config)
    return pipeline.run(video_path, output_dir)


def main() -> int:
    """Entry point."""
    import yaml

    config_path = PROJECT_ROOT / "configs" / "base_config.yaml"
    bench_config_path = PROJECT_ROOT / "configs" / "benchmark_config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bcfg = yaml.safe_load(bench_config_path.read_text(encoding="utf-8"))
    video_path = Path(cfg.get("video_path", ""))

    pipeline_config = PipelineConfig(
        device="auto",
        conf_threshold=bcfg.get("conf_threshold", 0.25),
        iou_threshold=bcfg.get("iou_threshold", 0.45),
        frame_skip=bcfg.get("frame_skip", 1),
        max_frames=bcfg.get("max_frames"),
        warmup_frames=bcfg.get("warmup_frames", 3),
    )
    try:
        summary = run(video_path, RESULTS_DIR, pipeline_config)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
