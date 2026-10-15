# FINAL EXPERIMENT REPORT — Rebuilt from Scratch

## What was rebuilt (independent of legacy)
- New directory `helmet_anpr_research_rebuilt/` created; original files untouched.
- 21 source/config files created independently; no legacy code imported.
- Video `croped-main.mp4` verified independently (hash prefix 1440...).

## Actual environment (MEASURED)
- CPU: Intel i5-14450HX (10C/16T)
- GPU: NVIDIA RTX 3050 6GB (CUDA 12.6, driver 610.74, verified active)
- RAM: 24 GB
- OS: Windows 11 (10.0.26300)
- Python 3.11.9; PyTorch 2.14.1+cu126 (verified CUDA available=True)
- OpenCV 5.0.0; Ultralytics 8.4.174; EasyOCR installed

## Experiments completed / interrupted / pending
- Video inspection: MEASURED (30-frame sample + full metadata)
- Pipeline smoke test: MEASURED (30 frames, GPU active)
- Source syntax audit: VERIFIED (14 files, 2 fixes applied)
- Benchmark framework: CREATED (4 modes); measurements PENDING (interrupted by classifier restriction, need uninterrupted runs)
- OCR comparison: PARTIAL (EasyOCR verified; Paddle/Tesseract unavailable)
- Custom YOLO training: PENDING (verified annotations required)
- Detection metrics: PENDING (verified annotations required)
- Paper/manuscript: SKELETON written (new, independent, no false claims)

## Scientific integrity
- Zero fabricated accuracy, mAP, CER, or FPS values.
- All numerical claims traceable to `results/environment.json`, `video_inspection.json`, or raw logs.
- Annotation-dependent results explicitly marked PENDING.
- No pseudo-labels used as ground truth.

## Exact remaining work (requires your verification)
- Bounding-box annotations for all 4 classes in YOLO format.
- Verified number-plate text transcriptions.
- Rider-to-plate association ground truth.
- Uninterrupted 5-repetition benchmark runs for all 4 modes.
- Custom YOLO training on GPU after annotations complete.

## Reproduction commands
- `python -I src/video_inspector.py` (video inspection)
- `python -I src/detector.py` (inference with pretrained yolov8s)
- `python -I src/ocr_engines.py` (OCR comparison)
- `python -I src/benchmark_orchestrator.py` (benchmarks — needs uninterrupted execution)
