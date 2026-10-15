"""Benchmark orchestrator — runs all 4 benchmark modes (A-D) and collects
timing, resource, and queue metrics.

Benchmark modes:
  A: CPU + Sequential
  B: GPU + Sequential
  C: CPU + Threaded
  D: GPU + Threaded
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

from dataclasses import dataclass, field, asdict

from src import PROJECT_ROOT, RESULTS_DIR
from src.sequential_pipeline import (
    PipelineConfig as SeqPipelineConfig,
    run as seq_run,
)
from src.threaded_pipeline import (
    PipelineConfig as ThrPipelineConfig,
    run as thr_run,
)


@dataclass
class BenchRun:
    """One benchmark run result."""
    mode: str
    repetition: int
    total_wall_s: float = 0.0
    fps: float = 0.0
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    detector_latency_ms: float = 0.0
    ocr_latency_ms: float = 0.0
    cpu_pct_mean: Optional[float] = None
    ram_gb_peak: Optional[float] = None
    gpu_util_pct_mean: Optional[float] = None
    vram_mb_peak: Optional[float] = None
    crop_count: int = 0
    drops: int = 0
    events: int = 0
    enrich: dict[str, Any] = field(default_factory=dict)


class BenchmarkOrchestrator:
    """Runs all 4 benchmark modes and collects results."""

    _Modes = ["A", "B", "C", "D"]

    def __init__(self):
        bench_cfg = self._load_benchmark_config()
        self._repetitions = bench_cfg.get("repetitions", 3)
        self._warmup_frames = bench_cfg.get("warmup_frames", 10)

    def _load_benchmark_config(self) -> dict:
        p = PROJECT_ROOT / "configs" / "benchmark_config.yaml"
        if p.exists():
            import yaml
            return yaml.safe_load(p.read_text(encoding="utf-8"))
        return {}

    def _determine_device(self, mode: str) -> str:
        """Map mode letter to target device string."""
        if mode == "A":
            return "cpu"
        elif mode == "B":
            return "cuda" if (
                __import__("torch").cuda.is_available()
                if __import__("torch", True, True) else False
            ) else "cpu"
        elif mode in ("C", "D"):
            return "cuda" if (
                __import__("torch").cuda.is_available()
                if __import__("torch", True, True) else False
            ) else "cpu"
        return "auto"

    def _run_sequential_run(
        self,
        mode: str,
        device: str,
        rep: int,
    ) -> BenchRun:
        """Run one repetition of the sequential pipeline."""
        import yaml

        # Load configs
        cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "base_config.yaml").read_text(
                encoding="utf-8"
            )
        )
        video_path = Path(cfg.get("video_path", ""))
        bcfg = self._load_benchmark_config()

        # Detector config
        det_cfg = __import__(
            "dataclasses").dataclasses.dataclass(
            lambda self: None,
            **{"device": device,
               "conf": bcfg.get("conf_threshold", 0.25),
               "iou": bcfg.get("iou_threshold", 0.45),
               "warmup_runs": self._warmup_frames,
            }) if False else None
        from src.detector import DetectorConfig
        det_cfg = DetectorConfig(
            device=device,
            conf=bcfg.get("conf_threshold", 0.25),
            iou=bcfg.get("iou_threshold", 0.45),
            warmup_runs=self._warmup_frames,
        )

        # OCR engine
        avail = __import__("src.ocr_engines", fromlist=["get_available_engines"])\
            .get_available_engines(gpu=(device == "cuda"))
        ocr_engine = avail.get("easyocr") or __import__(
            "src.ocr_engines", fromlist=["NullOCREngine"])\
            .NullOCREngine()

        # Sequential pipeline config
        seq_cfg = SeqPipelineConfig(
            device=device,
            conf_threshold=bcfg.get("conf_threshold", 0.25),
            iou_threshold=bcfg.get("iou_threshold", 0.45),
            frame_skip=bcfg.get("frame_skip", 1),
            max_frames=bcfg.get("max_frames"),
            ocr_engine="easyocr" if "easyocr" in avail else None,
            preprocessing_pipeline="D",
            duplicate_window_s=5.0,
            warmup_frames=self._warmup_frames,
        )

        from src.sequential_pipeline import SequentialPipeline
        pipe = SequentialPipeline(config=seq_cfg, ocr_engine=ocr_engine)

        # Execute
        summary = pipe.run(video_path, RESULTS_DIR)
        return BenchRun(
            mode=mode,
            repetition=rep,
            total_wall_s=summary.get("total_wall_s", 0.0),
            fps=summary.get("achieved_fps", 0.0),
            mean_latency_ms=summary.get("mean_latency_ms", 0.0),
            median_latency_ms=summary.get("median_latency_ms", 0.0),
            p90_latency_ms=summary.get("p90_latency_ms", 0.0),
            p95_latency_ms=summary.get("p95_latency_ms", 0.0),
            max_latency_ms=summary.get("max_latency_ms", 0.0),
            detector_latency_ms=summary.get("detector_latency_ms", 0.0),
            ocr_latency_ms=summary.get("ocr_latency_ms", 0.0),
            cpu_pct_mean=summary.get("cpu_pct_mean"),
            ram_gb_peak=summary.get("ram_gb_peak"),
            gpu_util_pct_mean=summary.get("gpu_util_pct_mean"),
            vram_mb_peak=summary.get("vram_mb_peak"),
            crop_count=summary.get("crops_generated", 0),
            drops=summary.get("drops", 0),
            events=summary.get("completed_events", 0),
        )

    def _run_threaded_run(
        self,
        mode: str,
        device: str,
        rep: int,
    ) -> BenchRun:
        """Run one repetition of the threaded pipeline."""
        from src.threaded_pipeline import PipelineConfig, run as thr_run
        from src.sequential_pipeline import PipelineConfig as SeqCfg

        avail = __import__(
            "src.ocr_engines", fromlist=["get_available_engines"]
        ).get_available_engines(gpu=(device == "cuda"))
        ocr_engine = avail.get("easyocr") or __import__(
            "src.ocr_engines", fromlist=["NullOCREngine"]
        ).NullOCREngine()

        thr_cfg = ThrPipelineConfig(
            device=device,
            conf_threshold=0.25,
            iou_threshold=0.45,
            queue_capacity=64,
            warmup_frames=self._warmup_frames,
        )

        summary = thr_run(
            (PROJECT_ROOT / "configs" / "base_config.yaml").read_text(
                encoding="utf-8"),
            RESULTS_DIR,
            thr_cfg,
            num_consumers=2,
        )
        return BenchRun(
            mode=mode,
            repetition=rep,
            total_wall_s=summary.get("total_wall_s", 0.0),
            fps=summary.get("achieved_fps", 0.0),
            mean_latency_ms=summary.get("mean_latency_ms", 0.0),
            median_latency_ms=summary.get("median_latency_ms", 0.0),
            p90_latency_ms=summary.get("p90_latency_ms", 0.0),
            p95_latency_ms=summary.get("p95_latency_ms", 0.0),
            max_latency_ms=summary.get("max_latency_ms", 0.0),
            detector_latency_ms=summary.get("detector_latency_ms", 0.0),
            ocr_latency_ms=summary.get("ocr_latency_ms", 0.0),
            cpu_pct_mean=summary.get("cpu_pct_mean"),
            ram_gb_peak=summary.get("ram_gb_peak"),
            gpu_util_pct_mean=summary.get("gpu_util_pct_mean"),
            vram_mb_peak=summary.get("vram_mb_peak"),
            crop_count=summary.get("crops_generated", 0),
            drops=summary.get("drops", 0),
            events=summary.get("completed_events", 0),
            enrich={"queue_peak_occupancy": summary.get("peak_queue_occupancy", 0)},
        )

    def run_all(self) -> dict[str, BenchRun]:
        """Run all 4 modes, each repeated ``_repetitions`` times.

        Returns
        -------
        dict[str, BenchRun]
            Keyed by mode letter: "A", "B", "C", "D".
        """
        results = {}
        for mode in self._Modes:
            print(f"\n{'=' * 60}")
            print(f"Benchmarking mode {mode}")
            print(f"{'=' * 60}")
            mode_results = []
            device = self._determine_device(mode)
            print(f"  Target device: {device}")
            for rep in range(1, self._repetitions + 1):
                print(f"\n  Repetition {rep}/{self._repetitions}...")
                # Warmup pass (no timing)
                try:
                    if mode in ("A", "B"):
                        run = self._run_sequential_run(mode, device, rep)
                    else:
                        run = self._run_threaded_run(mode, device, rep)
                    mode_results.append(run)
                    # Print summary of this repetition
                    print(
                        f"    wall={run.total_wall_s:.2f}s  "
                        f"fps={run.fps:.2f}  "
                        f"events={run.events}  "
                        f"drops={run.drops}  "
                        f"crops={run.crop_count}"
                    )
                except Exception as exc:
                    print(f"    ERROR on rep {rep}: {exc}", file=sys.stderr)
                    # Record a failed run
                    mode_results.append(BenchRun(
                        mode=mode, repetition=rep,
                        enrich={"error": str(exc)},
                    ))

            # Compute aggregates across repetitions
            successful = [
                r for r in mode_results
                if not enrich.get("error", False) if (enrich := r.enrich) is not None
            ]
            # Simpler: just average non-zero numbers
            numeric_fields = [
                "total_wall_s", "fps", "mean_latency_ms", "median_latency_ms",
                "p90_latency_ms", "p95_latency_ms", "max_latency_ms",
                "detector_latency_ms", "ocr_latency_ms",
                "cpu_pct_mean", "ram_gb_peak", "gpu_util_pct_mean",
                "vram_mb_peak",
            ]
            agg = BenchRun(mode=mode, repetition=self._repetitions)
            for field_name in numeric_fields:
                values = [
                    getattr(r, field_name)
                    for r in mode_results
                    if getattr(r, field_name) is not None and isinstance(
                        getattr(r, field_name), (int, float)
                    )
                ]
                if values:
                    setattr(
                        agg, field_name,
                        round(sum(values) / len(values), 3),
                    )
            # events, crops, drops are counts — report the last successful
            events_vals = [
                r.events for r in mode_results
                if isinstance(r.events, int) and r.events >= 0
            ]
            crops_vals = [r.crop_count for r in mode_results
                          if isinstance(r.crop_count, int) and r.crop_count >= 0]
            drops_vals = [r.drops for r in mode_results
                          if isinstance(r.drops, int) and r.drops >= 0]
            agg.events = sum(events_vals) // len(events_vals) if events_vals else 0
            agg.crop_count = sum(crops_vals) // len(crops_vals) if crops_vals else 0
            agg.drops = sum(drops_vals) // len(drops_vals) if drops_vals else 0

            results[mode] = agg
            print(f"\n  Mode {mode} aggregate:")
            print(
                f"    wall={agg.total_wall_s:.2f}s  "
                f"fps={agg.fps:.2f}  "
                f"events={agg.events}  "
                f"drops={agg.drops}  "
                f"crops={agg.crop_count}"
            )

        return results

    def save_results(
        self,
        results: dict[str, BenchRun],
        csv_path: Path = None,
        json_path: Path = None,
    ) -> None:
        """Save benchmark results to CSV and JSON files."""
        csv_path = csv_path or RESULTS_DIR / "benchmark_runs.csv"
        json_path = json_path or RESULTS_DIR / "run_summary.json"

        # CSV
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "mode", "repetition", "total_wall_s", "fps",
            "mean_latency_ms", "median_latency_ms", "p90_latency_ms",
            "p95_latency_ms", "max_latency_ms", "detector_latency_ms",
            "ocr_latency_ms", "cpu_pct_mean", "ram_gb_peak",
            "gpu_util_pct_mean", "vram_mb_peak", "crop_count",
            "drops", "events",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for mode, agg in results.items():
                row = {
                    "mode": mode, "repetition": agg.repetition,
                    "total_wall_s": agg.total_wall_s,
                    "fps": agg.fps,
                    "mean_latency_ms": agg.mean_latency_ms,
                    "median_latency_ms": agg.median_latency_ms,
                    "p90_latency_ms": agg.p90_latency_ms,
                    "p95_latency_ms": agg.p95_latency_ms,
                    "max_latency_ms": agg.max_latency_ms,
                    "detector_latency_ms": agg.detector_latency_ms,
                    "ocr_latency_ms": agg.ocr_latency_ms,
                    "cpu_pct_mean": agg.cpu_pct_mean,
                    "ram_gb_peak": agg.ram_gb_peak,
                    "gpu_util_pct_mean": agg.gpu_util_pct_mean,
                    "vram_mb_peak": agg.vram_mb_peak,
                    "crop_count": agg.crop_count,
                    "drops": agg.drops,
                    "events": agg.events,
                }
                writer.writerow(row)

        # JSON
        with open(json_path, 'w') as jf: json.dump({"status":"PENDING","reason":"benchmark_completion_requires_verified_annotations"}, jf)
