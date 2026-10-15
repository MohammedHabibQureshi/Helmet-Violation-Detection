"""Background resource monitor.

Samples CPU%, RAM, and GPU metrics every 0.5 seconds in a background thread.
Provides a context manager interface and per-metric summary statistics.
"""

import csv
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src import RESULTS_DIR

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

# Try to import pynvml for GPU monitoring
try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML_AVAILABLE = True
except Exception:
    pynvml = None
    _NVML_HANDLE = None
    _NVML_AVAILABLE = False


@dataclass
class ResourceSample:
    """One sample of resource usage."""

    timestamp: float
    cpu_pct: Optional[float]
    ram_used_gb: Optional[float]
    ram_pct: Optional[float]
    gpu_util_pct: Optional[float]
    vram_used_mb: Optional[float]


class ResourceMonitor:
    """Background resource monitor.

    Samples CPU%, RAM, and GPU metrics every ``interval_s`` seconds.

    Usage as context manager::

        with ResourceMonitor() as mon:
            ... do work ...
        summary = mon.get_summary()

    Or manually::

        mon = ResourceMonitor()
        mon.start()
        ... do work ...
        mon.stop()
        summary = mon.get_summary()
    """

    def __init__(
        self,
        interval_s: float = 0.5,
        output_path: Optional[Path] = None,
    ):
        self.interval_s = interval_s
        self.output_path = output_path or RESULTS_DIR / "resource_metrics.csv"
        self._samples: list[ResourceSample] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def _sample_once(self) -> ResourceSample:
        """Take a single resource snapshot."""
        ts = time.time()
        cpu_pct = None
        ram_used_gb = None
        ram_pct = None
        gpu_util_pct = None
        vram_used_mb = None

        if _PSUTIL_AVAILABLE:
            try:
                cpu_pct = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                ram_used_gb = round(mem.used / (1024 ** 3), 3)
                ram_pct = round(mem.percent, 2)
            except Exception:
                pass

        if _NVML_AVAILABLE and pynvml is not None and _NVML_HANDLE is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                gpu_util_pct = float(util.gpu)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                vram_used_mb = round(mem_info.used / (1024 ** 2), 1)
            except Exception:
                pass

        return ResourceSample(
            timestamp=ts,
            cpu_pct=cpu_pct,
            ram_used_gb=ram_used_gb,
            ram_pct=ram_pct,
            gpu_util_pct=gpu_util_pct,
            vram_used_mb=vram_used_mb,
        )

    def _monitor_loop(self) -> None:
        """Background sampling loop."""
        # Prime CPU% measurement (first call returns 0.0 by design in psutil)
        if _PSUTIL_AVAILABLE:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass
        while not self._stop_event.is_set():
            sample = self._sample_once()
            with self._lock:
                self._samples.append(sample)
            self._stop_event.wait(timeout=self.interval_s)

    def start(self) -> None:
        """Start background sampling."""
        if self._started:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="resource-monitor"
        )
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        """Stop background sampling and save to CSV."""
        if not self._started:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._save_csv()
        self._started = False

    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def _save_csv(self) -> None:
        """Save samples to CSV."""
        path = self.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            samples = list(self._samples)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "timestamp", "cpu_pct", "ram_used_gb", "ram_pct",
                    "gpu_util_pct", "vram_used_mb",
                ],
            )
            writer.writeheader()
            for s in samples:
                writer.writerow({
                    "timestamp": round(s.timestamp, 3),
                    "cpu_pct": s.cpu_pct,
                    "ram_used_gb": s.ram_used_gb,
                    "ram_pct": s.ram_pct,
                    "gpu_util_pct": s.gpu_util_pct,
                    "vram_used_mb": s.vram_used_mb,
                })

    def get_samples(self) -> list[ResourceSample]:
        """Return a copy of all collected samples."""
        with self._lock:
            return list(self._samples)

    def get_summary(self) -> dict:
        """Return peak/mean/min statistics for each metric."""
        with self._lock:
            samples = list(self._samples)

        def _stats(vals):
            clean = [v for v in vals if v is not None]
            if not clean:
                return {"mean": None, "peak": None, "min": None}
            return {
                "mean": round(sum(clean) / len(clean), 3),
                "peak": round(max(clean), 3),
                "min": round(min(clean), 3),
            }

        return {
            "sample_count": len(samples),
            "cpu_pct": _stats([s.cpu_pct for s in samples]),
            "ram_used_gb": _stats([s.ram_used_gb for s in samples]),
            "ram_pct": _stats([s.ram_pct for s in samples]),
            "gpu_util_pct": _stats([s.gpu_util_pct for s in samples]),
            "vram_used_mb": _stats([s.vram_used_mb for s in samples]),
            "psutil_available": _PSUTIL_AVAILABLE,
            "nvml_available": _NVML_AVAILABLE,
        }


def main() -> int:
    """Smoke-test: monitor resources for 3 seconds."""
    print("Starting resource monitor for 3 seconds...")
    with ResourceMonitor(interval_s=0.5) as mon:
        time.sleep(3.0)
    summary = mon.get_summary()
    print(f"Samples collected: {summary['sample_count']}")
    print(f"CPU%:    mean={summary['cpu_pct']['mean']}  "
          f"peak={summary['cpu_pct']['peak']}")
    print(f"RAM GB:  mean={summary['ram_used_gb']['mean']}  "
          f"peak={summary['ram_used_gb']['peak']}")
    print(f"GPU%:    mean={summary['gpu_util_pct']['mean']}  "
          f"peak={summary['gpu_util_pct']['peak']}")
    print(f"VRAM MB: mean={summary['vram_used_mb']['mean']}  "
          f"peak={summary['vram_used_mb']['peak']}")
    print(f"CSV:     {mon.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
