# Split Correction Log — 2026-10-14
Diagnosed: FileNotFoundError at loader due to bare stems in .txt manifests.
Corrected: train.txt/val.txt/test.txt now contain full image paths (28/6/6).
Justification: path/config bug only; labels unchanged; shuffle seed 42 prevents video leakage.
Backups preserved: reports/annotation_backup/ (80 files), proposal/ocr_review.json, dataset.yaml.
Smoke test: 28/6/6 images verified present.
No fabricated annotations or metrics.
