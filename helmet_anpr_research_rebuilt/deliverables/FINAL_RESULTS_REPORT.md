# FINAL RESULTS REPORT — Helmet Violation Detection + ANPR
Date: 2026-10-15
Project root: helmet_anpr_research_rebuilt/

## Commands executed (reproducible)
- python -I scripts/diagnose_environment.py
- python -I scripts/test_cuda.py
- PYTHONPATH=src python -c (detector smoke, association run, CSV/JSON write)
- python (dataset audit: count images/labels/boxes/classes/empty)
- PYTHONPATH=src python (benchmark CPU vs CUDA, 5 repetitions each)
- python (Table III / VIII generation from saved JSON/CSV)

## Actual measured values
- CUDA available: True; device = NVIDIA GeForce RTX 3050 6GB; torch 2.14.1+cu126
- Dataset: 40 frames; 101 boxes; 6 empty; 28/6/6 split
- Custom best.pt: 4 classes (verified via YOLO names); loads; detects with_helmet on test frame
- Association links (recomputed): 27; independent reference: NO → precision/recall/F1 undefined
- OCR (EasyOCR GPU): executed; reference transcriptions unavailable → CER/EMN undefined
- Runtime CPU: mean ~150ms/frame; CUDA: mean ~15ms/frame (5 reps each, synchronized)
- End-to-end: not measured (requires verified pipeline + reference events)

## Tests passed / failed
- ENV/CUDA diagnostics: PASS
- Detector load + inference (custom weights): PASS
- Dataset validation (structural): PASS (semantic review deferred)
- Association (stable IDs): PASS (evaluation deferred)
- Benchmark (CPU/CUDA): PASS (values recorded)
- OCR: PARTIAL (engine runs; evaluation blocked by missing GT)
- Table generation: PASS (derived from saved files)

## Remaining dependencies
- Independent annotation review (40 frames) needed before detection/AP/F1 can be reported.
- Independent plate-transcription review needed before OCR CER/EMN can be reported.
- Independent association adjudication needed before P/R/F1 can be reported.
- Full end-to-end video pipeline requires verified labels.
No fabricated metrics. All unavailable metrics reported with exact reason.
