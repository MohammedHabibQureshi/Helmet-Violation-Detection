"""YOLOv8 detector wrapper for the Helmet Violation Detection project.

Loads a YOLOv8 model (custom weights if present, else pretrained) and wraps
inference into a simple per-frame API.
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src import PROJECT_ROOT

# Class mapping for the 4 project classes
CLASS_MAP = {
    0: "rider",
    1: "with_helmet",
    2: "without_helmet",
    3: "number_plate",
}

DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"
FALLBACK_WEIGHTS = "yolov8s.pt"


@dataclass
class DetectorConfig:
    """Configuration for the YOLODetector."""

    weights_path: Optional[Path] = None
    device: str = "auto"              # 'auto', 'cpu', 'cuda', 'cuda:0'
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    warmup_runs: int = 3
    # Whether the loaded model uses the project's custom class mapping
    use_custom_classes: bool = True


@dataclass
class DetectionResult:
    """Single detection from one frame."""

    class_id: int
    class_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    area: float


class YOLODetector:
    """Wraps a YOLOv8 model for single-frame inference.

    Parameters
    ----------
    config:
        Detector configuration. If config.weights_path is None, tries
        ``weights/best.pt`` from the project root; falls back to the
        pretrained ``yolov8s.pt`` from Ultralytics.
    """

    def __init__(self, config: DetectorConfig = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLODetector. "
                "Install it with: pip install ultralytics"
            ) from exc

        self.config = config or DetectorConfig()
        self._using_custom = False

        # Resolve weights
        weights = self.config.weights_path
        if weights is None:
            weights = DEFAULT_WEIGHTS
        weights = Path(weights)

        if weights.exists():
            self._model = YOLO(str(weights))
            self._using_custom = True
            self._weights_note = f"Custom weights: {weights}"
        else:
            self._model = YOLO(FALLBACK_WEIGHTS)
            self._using_custom = False
            self._weights_note = (
                f"Custom weights not found at {weights}; "
                f"using pretrained {FALLBACK_WEIGHTS}"
            )
            print(f"[WARNING] {self._weights_note}", file=sys.stderr)

        # Resolve device
        device = self.config.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self._device = device

    @property
    def using_custom_weights(self) -> bool:
        """True if custom weights were found and loaded."""
        return self._using_custom

    @property
    def weights_note(self) -> str:
        """Human-readable note about which weights are loaded."""
        return self._weights_note

    def warmup(self, n: int = None) -> None:
        """Run inference on a blank frame to warm up the model."""
        runs = n if n is not None else self.config.warmup_runs
        blank = np.zeros((self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8)
        for _ in range(runs):
            self._model.predict(
                blank,
                conf=self.config.conf,
                iou=self.config.iou,
                device=self._device,
                verbose=False,
            )

    def detect_frame(
        self,
        frame: np.ndarray,
        conf: float = None,
        iou: float = None,
    ) -> list[dict]:
        """Run inference on a single frame.

        Parameters
        ----------
        frame:
            BGR image array (as returned by cv2.imread / cap.read).
        conf:
            Confidence threshold (overrides config).
        iou:
            IOU threshold (overrides config).

        Returns
        -------
        list of dict
            Each dict has keys: class_id, class_name, conf, x1, y1, x2, y2,
            area, inference_ms.
        """
        conf = conf if conf is not None else self.config.conf
        iou = iou if iou is not None else self.config.iou

        t0 = time.perf_counter()
        results = self._model.predict(
            frame,
            conf=conf,
            iou=iou,
            device=self._device,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)

                # Map class name from custom classes or COCO
                if self._using_custom and self.config.use_custom_classes:
                    class_name = CLASS_MAP.get(class_id, f"class_{class_id}")
                else:
                    # Fall back to model names
                    try:
                        class_name = self._model.names.get(
                            class_id, f"class_{class_id}"
                        )
                    except Exception:
                        class_name = f"class_{class_id}"

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "conf": round(confidence, 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "area": round(area, 2),
                    "inference_ms": round(inference_ms, 3),
                })
        return detections


def get_detector(device: str = "auto", conf: float = 0.25) -> YOLODetector:
    """Convenience factory for a default YOLODetector."""
    cfg = DetectorConfig(device=device, conf=conf)
    return YOLODetector(cfg)


def main() -> int:
    """Entry point — smoke-test the detector on a blank frame."""
    import numpy as np

    print("Loading detector...")
    try:
        detector = YOLODetector()
        print(f"Weights note: {detector.weights_note}")
        print("Warming up...")
        detector.warmup()
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        print("Running inference on blank frame...")
        detections = detector.detect_frame(blank)
        print(f"Detections on blank frame: {len(detections)}")
        print("Detector OK.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
