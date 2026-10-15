# OCR RESULTS — Actual (Not Fabricated)
Date: 2026-10-14
Status: PARTIAL — predictions preserved; no verified ground truth available

## Engines
- EasyOCR: works (used for 25 candidates in proposal/ocr_review.json).
- PaddleOCR: broken (import failure due numpy/sklearn binary mismatch — not fixed; documented in audit).

## Candidates
- Existing: 25 unverified EasyOCR predictions (proposal/ocr_review.json).
- Verified plate text: NONE — user visual verification of plate crops required before any OCR accuracy can be calculated.

## Accuracy metrics
- NOT COMPUTED — requires verified transcription ground truth (user task, blocked).
- Not fabricated — explicitly stated as unavailable.

## Benchmark (measured, minimal)
File: results/benchmarks/ocr_benchmark.csv
- easyocr single CPU 1 crop ~1520 ms (measured, not averaged over repeats — documented limitation).
- Multithreaded/sequential CPU/GPU comparison: NOT COMPLETED (single crop, single run, documented as minimal).

## Separation
- Predictions kept in proposal/ocr_review.json separately from annotations.
- Not merged into data/labels/ or associations.
