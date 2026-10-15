"""Helmet Violation Detection + ANPR research project.

A modular pipeline for detecting helmet violations on motorcyclists and
associating them with vehicle number plates via OCR.

Modules:
    - video_inspector: inspect video properties, motion blur, brightness
    - frame_extractor: extract frames from video
    - detector: YOLOv8 detector wrapper
    - helmet_association: link helmetless riders to number plates
    - plate_preprocessor: plate image preprocessing pipelines
    - ocr_engines: EasyOCR, PaddleOCR, Tesseract wrappers
    - duplicate_suppressor: deduplicate violation events
    - sequential_pipeline: single-threaded video processing
    - threaded_pipeline: producer-consumer multithreaded processing
    - resource_monitor: CPU/RAM/GPU monitoring
    - benchmark_orchestrator: run A/B/C/D benchmark modes
    - detection_evaluator: evaluation metrics (mAP, CER)
    - error_analyzer: analyze annotation/inference errors
    - statistical_analyzer: statistics over runs
    - figure_generator: plot figures
    - table_generator: export tables
"""

from pathlib import Path

__version__ = "0.1.0"
__project__ = "Helmet-Violation-Detection"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

__all__ = [
    "PROJECT_ROOT",
    "CONFIGS_DIR",
    "DATA_DIR",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "OUTPUTS_DIR",
    "__version__",
    "__project__",
]
