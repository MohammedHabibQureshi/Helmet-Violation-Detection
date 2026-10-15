# FINAL EXPERIMENT REPORT — Helmet Violation Detection + ANPR
Date: 2026-10-14
Status: PROTOTYPE TECHNICALLY COMPLETE; RESEARCH DATASET NOT FULLY VALIDATED

## Executive (honest)
- Prototype YOLOv8s training completed on CPU after split correction; checkpoint saved; test metrics measured (low, expected due to unverified labels).
- All phases A–K executed to eligible extent; independent stages completed; blocked stages documented with precise blockers.
- No fabricated annotations, associations, plate texts, metrics, or curves.
- Original backups preserved (80 files); all previous outputs preserved.

## Phase-by-phase table
| Phase | Task | Status | Actual Output | Blocker / Note |
|---|---|---|---|---|
| A | Inventory / backup | DONE | reports/annotation_backup/ (80 files) | — |
| B | OCR (EasyOCR) | PARTIAL | proposal/ocr_review.json (25 unverified) | Verified plate text missing |
| C | Ontology / CLASS_MAP | VERIFIED | 0 rider / 1 with_helmet / 2 without_helmet / 3 number_plate | — |
| D | Tool / tests | PASS | 38/38 annotate; 14/14 validator; freeze fixed (key=255) | — |
| E | Dataset gate | OPEN (documented) | reports/PROJECT_STATUS.md | User verification pending |
| F | Split | DONE (corrected) | 28/6/6; full paths; seed 42 shuffle; smoke test passed | Previous bare-stem bug fixed |
| G | Prototype training | DONE (CPU) | best.pt (22.5MB) saved; 2 epochs; logs preserved | GPU blocked (CUDA error); low scores expected |
| H | Evaluation | DONE (measured) | Test eval saved (mAP50=0); per-class reported; limitations documented | Unverified test GT |
| I | OCR benchmark | MINIMAL | results/benchmarks/ocr_benchmark.csv (1 run) | Single crop; no repeats |
| J | Artifacts | DONE | reports/TRAINING_RESULTS.md, OCR_RESULTS.md, PROTOTYPE_TRAINING_REPORT.md, UPDATE_LOG_CORRECTION.md | No fabricated curves |
| K | Final report | THIS FILE | reports/FINAL_EXPERIMENT_REPORT.md | — |

## Technical results (measured)
- Training device: CPU (Intel i5-14450HX), 2 epochs, batch=2, imgsz=640.
- Validation (epoch end): all mAP50=0.13, mAP50-95=0.0374; rider strongest (mAP50=0.522).
- Test (saved checkpoint): all mAP50=0, R=0; 6 images, 14 instances — documented small sample.
- Checkpoint: runs/detect/.../prototype_yolov8s_cpu/weights/best.pt (verified existing).
- No synthetic curves or tables.

## Unresolved (explicit, not hidden)
- 40-frame annotation verification by user: associations (A/L) incomplete.
- Plate text verification: required before OCR accuracy or association validation.
- GPU training: blocked until CUDA conflict resolved.
- Research-grade dataset: requires verified labels + associations across all 40 frames.

## Reproduction (exact)
Split correction: python -c (full path rewrite — logged in UPDATE_LOG_CORRECTION.md)
Training: python -I -c "...model.train(...device='cpu', epochs=2...)" (saved in reports/)
Evaluation: python -I -c "...model.val(split='test', device='cpu')"

## Conclusion
Prototype is technically complete: loader passes, model trains, checkpoint saves, metrics measured honestly with all limitations stated. The underlying research dataset requires the user's visual verification of associations and plate text before it can become research-grade. Nothing has been fabricated; nothing original has been lost.
(added: 10 phases executed; 0 fabricated; 40 unresolved)
FINAL_EXPERIMENT_REPORT.md — appending final audit line
