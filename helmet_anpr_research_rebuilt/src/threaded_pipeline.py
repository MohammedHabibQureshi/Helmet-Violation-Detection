"""Producer-consumer multithreaded video processing pipeline.

Producer thread: reads frames, runs detection, enqueues plate crops.
Consumer thread(s): dequeue crops, run OCR, suppress duplicates.
Uses a bounded FIFO queue with DROP_OLDEST overflow policy.
"""

import csv
import json
import queue
import sys
import threading
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
from src.sequential_pipeline import PipelineConfig, _save_events_csv


# Sentinel value to signal consumer shutdown
_SENTINEL = object()


@dataclass
class CropItem:
    """A plate crop waiting for OCR in the consumer queue."""

    frame_index: int
    timestamp_s: float
    crop: np.ndarray
    preprocessing_pipeline: str
    candidate: dict
    enqueue_time: float = field(default_factory=time.perf_counter)


class ThreadedPipeline:
    """Producer-consumer pipeline with bounded queue and DROP_OLDEST overflow.

    Parameters
    ----------
    config:
        Pipeline configuration (same dataclass as SequentialPipeline).
    ocr_engine:
        Pre-instantiated OCR engine. Resolved from config if None.
    num_consumers:
        Number of consumer (OCR) worker threads.
    """

    def __init__(
        self,
        config: PipelineConfig = None,
        ocr_engine=None,
        num_consumers: int = 1,
    ):
        self.config = config or PipelineConfig()
        cfg = self.config
        self._num_consumers = num_consumers

        # Detector (owned by producer thread)
        self._detector = YOLODetector(
            DetectorConfig(
                device=cfg.device,
                conf=cfg.conf_threshold,
                iou=cfg.iou_threshold,
                warmup_runs=cfg.warmup_frames,
            )
        )

        # OCR engine (shared by consumer threads via lock)
        if ocr_engine is not None:
            self._ocr = ocr_engine
        else:
            available = get_available_engines()
            self._ocr = available.get(cfg.ocr_engine) or NullOCREngine()
        self._ocr_lock = threading.Lock()

        self._preprocessor = PlatePreprocessor()
        self._suppressor = DuplicateSuppressor(
            window_seconds=cfg.duplicate_window_s
        )
        self._suppressor_lock = threading.Lock()
        self._assoc_config = AssociationConfig()

        # Bounded queue
        self._queue: queue.Queue = queue.Queue(
            maxsize=cfg.queue_capacity
        )

        # Thread-safe counters
        self._counter_lock = threading.Lock()
        self._counters = {
            "frames_read": 0,
            "frames_processed": 0,
            "frames_skipped": 0,
            "crops_generated": 0,
            "crops_enqueued": 0,
            "crops_dequeued": 0,
            "crops_processed": 0,
            "crops_rejected": 0,
            "queue_full_events": 0,
            "crops_dropped": 0,
            "worker_exceptions": 0,
            "duplicate_suppressed": 0,
            "completed_events": 0,
            "peak_queue_occupancy": 0,
        }
        self._events: list[dict] = []
        self._events_lock = threading.Lock()

        # Per-event latency tracking
        self._latencies: list[float] = []
        self._latency_lock = threading.Lock()

        # Exception propagation
        self._producer_exc: Optional[Exception] = None
        self._consumer_exc: Optional[Exception] = None

    def _inc(self, key: str, amount: int = 1) -> None:
        with self._counter_lock:
            self._counters[key] += amount

    def _update_peak_queue(self) -> None:
        with self._counter_lock:
            qsize = self._queue.qsize()
            if qsize > self._counters["peak_queue_occupancy"]:
                self._counters["peak_queue_occupancy"] = qsize

    def _drop_oldest(self) -> None:
        """Drop the oldest item from the queue to make room."""
        try:
            self._queue.get_nowait()
            self._inc("crops_dropped")
            self._inc("queue_full_events")
        except queue.Empty:
            pass

    def _producer_fn(
        self,
        video_path: Path,
        fps: float,
    ) -> None:
        """Producer thread: reads frames, detects, enqueues crops."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {video_path}")
            frame_index = 0
            cfg = self.config
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    self._inc("frames_read")

                    if frame_index % cfg.frame_skip != 0:
                        self._inc("frames_skipped")
                        frame_index += 1
                        continue

                    timestamp_s = frame_index / fps if fps > 0 else 0.0

                    # Detect
                    detections = self._detector.detect_frame(
                        frame, conf=cfg.conf_threshold, iou=cfg.iou_threshold
                    )
                    self._inc("frames_processed")

                    # Associate
                    candidates = associate_violations(
                        detections, frame_index, timestamp_s, self._assoc_config
                    )
                    accepted = [c for c in candidates if c["accepted"]]

                    for candidate in accepted:
                        plate_box = candidate.get("plate_box")
                        if plate_box is None:
                            self._inc("crops_rejected")
                            continue
                        crop = self._preprocessor.extract_plate_crop(
                            frame, plate_box
                        )
                        if crop is None:
                            self._inc("crops_rejected")
                            continue

                        self._inc("crops_generated")
                        item = CropItem(
                            frame_index=frame_index,
                            timestamp_s=timestamp_s,
                            crop=crop,
                            preprocessing_pipeline=cfg.preprocessing_pipeline,
                            candidate=candidate,
                            enqueue_time=time.perf_counter(),
                        )

                        # DROP_OLDEST overflow policy
                        if self._queue.full():
                            self._drop_oldest()

                        self._queue.put(item)
                        self._inc("crops_enqueued")
                        self._update_peak_queue()

                    if (
                        cfg.max_frames is not None
                        and self._counters["frames_processed"] >= cfg.max_frames
                    ):
                        break
                    frame_index += 1
            finally:
                cap.release()
        except Exception as exc:
            self._producer_exc = exc
        finally:
            # Send one sentinel per consumer
            for _ in range(self._num_consumers):
                self._queue.put(_SENTINEL)

    def _consumer_fn(self) -> None:
        """Consumer thread: dequeues crops, runs OCR, suppresses duplicates."""
        try:
            while True:
                try:
                    item = self._queue.get(timeout=2.0)
                except queue.Empty:
                    continue

                if item is _SENTINEL:
                    break

                self._inc("crops_dequeued")

                try:
                    dequeue_time = time.perf_counter()
                    pipeline_result = self._preprocessor.run_pipeline(
                        item.crop, item.preprocessing_pipeline
                    )
                    processed_image = pipeline_result.get("image")

                    if processed_image is None:
                        self._inc("crops_rejected")
                        self._queue.task_done() if hasattr(
                            self._queue, "task_done"
                        ) else None
                        continue

                    with self._ocr_lock:
                        ocr_result: OCRResult = self._ocr.recognize(
                            processed_image
                        )
                    self._inc("crops_processed")

                    plate_text = (
                        ocr_result.text_corrected
                        or ocr_result.text_normalized
                    )
                    if not plate_text:
                        self._inc("crops_rejected")
                        continue

                    with self._suppressor_lock:
                        is_dup = self._suppressor.is_duplicate(
                            plate_text, item.timestamp_s
                        )

                    if is_dup:
                        self._inc("duplicate_suppressed")
                        continue

                    # Record completed event
                    self._inc("completed_events")
                    total_latency_ms = (
                        (time.perf_counter() - item.enqueue_time) * 1000
                    )
                    with self._latency_lock:
                        self._latencies.append(total_latency_ms)
                    with self._events_lock:
                        self._events.append({
                            "frame_index": item.frame_index,
                            "timestamp_s": round(item.timestamp_s, 4),
                            "plate_text_raw": ocr_result.text_raw,
                            "plate_text_normalized": ocr_result.text_normalized,
                            "plate_text_corrected": plate_text,
                            "ocr_confidence": ocr_result.confidence,
                            "ocr_engine": ocr_result.engine_name,
                            "preprocessing_pipeline": item.preprocessing_pipeline,
                            "plate_box": item.candidate.get("plate_box"),
                            "queue_latency_ms": round(total_latency_ms, 3),
                        })
                except Exception as exc:
                    self._inc("worker_exceptions")
                    self._consumer_exc = exc
        except Exception as exc:
            self._consumer_exc = exc

    def _warmup(self) -> None:
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(self.config.warmup_frames):
            self._detector.detect_frame(blank)

    def run(
        self,
        video_path: str | Path,
        output_dir: str | Path = None,
    ) -> dict:
        """Run the threaded pipeline on a video.

        Parameters
        ----------
        video_path:
            Path to the video file.
        output_dir:
            Directory for result files.

        Returns
        -------
        dict
            Run summary.
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
        cap.release()

        print(f"[ThreadedPipeline] Warmup ({self.config.warmup_frames} frames)...")
        self._warmup()

        print(f"[ThreadedPipeline] Starting pipeline: "
              f"{self._num_consumers} consumer(s), "
              f"queue_capacity={self.config.queue_capacity}")

        wall_start = time.perf_counter()

        # Start consumer threads
        consumers = [
            threading.Thread(target=self._consumer_fn, daemon=True, name=f"consumer-{i}")
            for i in range(self._num_consumers)
        ]
        for t in consumers:
            t.start()

        # Run producer in main thread context
        producer_thread = threading.Thread(
            target=self._producer_fn, args=(video_path, fps),
            daemon=False, name="producer"
        )
        producer_thread.start()
        producer_thread.join()

        # Wait for consumers to drain the queue
        for t in consumers:
            t.join()

        wall_s = time.perf_counter() - wall_start

        # Propagate exceptions
        if self._producer_exc:
            raise self._producer_exc
        if self._consumer_exc:
            raise self._consumer_exc

        # Counters reconciliation check
        c = self._counters
        assert (
            c["crops_generated"] == c["crops_enqueued"] + c["crops_dropped"]
        ), (
            f"Queue accounting mismatch: generated={c['crops_generated']} "
            f"enqueued={c['crops_enqueued']} dropped={c['crops_dropped']}"
        )

        processed = c["frames_processed"]
        achieved_fps = processed / wall_s if wall_s > 0 else 0.0

        latencies = self._latencies
        import statistics as _stats
        mean_latency = _stats.mean(latencies) if latencies else 0.0
        median_latency = _stats.median(latencies) if latencies else 0.0
        sorted_lat = sorted(latencies) if latencies else []
        p90 = sorted_lat[int(0.90 * len(sorted_lat))] if sorted_lat else 0.0
        p95 = sorted_lat[int(0.95 * len(sorted_lat))] if sorted_lat else 0.0
        max_latency = max(latencies) if latencies else 0.0

        summary = {
            "mode": "threaded",
            "device": self.config.device,
            "num_consumers": self._num_consumers,
            "ocr_engine": self.config.ocr_engine,
            "preprocessing_pipeline": self.config.preprocessing_pipeline,
            "queue_capacity": self.config.queue_capacity,
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

        # Save results
        events_csv = output_dir / "threaded_events.csv"
        _save_events_csv(self._events, events_csv)

        summary_path = output_dir / "threaded_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)

        # Also save a simple per-event latency CSV
        lat_csv = output_dir / "threaded_results.csv"
        with lat_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["event_index", "queue_latency_ms"])
            for idx, lat in enumerate(self._latencies):
                w.writerow([idx, round(lat, 3)])

        print(f"[ThreadedPipeline] Done: {processed} frames in {wall_s:.2f}s "
              f"({achieved_fps:.2f} fps)")
        print(f"  Drops: {c['crops_dropped']}  "
              f"Events: {c['completed_events']}  "
              f"Summary: {summary_path}")
        return summary


def run(
    video_path: str | Path,
    output_dir: str | Path = None,
    config: PipelineConfig = None,
    num_consumers: int = 1,
) -> dict:
    """Module-level convenience function."""
    pipeline = ThreadedPipeline(config=config, num_consumers=num_consumers)
    return pipeline.run(video_path, output_dir)


def main() -> int:
    """Entry point."""
    import yaml

    config_path = PROJECT_ROOT / "configs" / "base_config.yaml"
    bench_path = PROJECT_ROOT / "configs" / "benchmark_config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bcfg = yaml.safe_load(bench_path.read_text(encoding="utf-8"))
    video_path = Path(cfg.get("video_path", ""))

    pipeline_config = PipelineConfig(
        device="auto",
        conf_threshold=bcfg.get("conf_threshold", 0.25),
        iou_threshold=bcfg.get("iou_threshold", 0.45),
        frame_skip=bcfg.get("frame_skip", 1),
        max_frames=bcfg.get("max_frames"),
        queue_capacity=bcfg.get("queue_capacity", 64),
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
