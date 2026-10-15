# Dataset remediation report -- 2026-10-14

## Repairs completed

1. **Class-ID persistence defect repaired.** Every non-empty YOLO label used IDs 1--4 although the dataset ontology is 0--3. The defect was systematic (+1), confirmed against `reports/annotation_backup/`, and corrected as `1->0`, `2->1`, `3->2`, `4->3`. The hash-level change record is `LABEL_CLASS_SHIFT_REPAIR.json`.
2. **Structural validation rerun.** The repaired corpus contains 40 label files and 101 records; the validator reports zero format errors and zero warnings.
3. **Legacy OCR corpus rejected.** All 32 files in `outputs/review/plate_crops/` were visually audited and do not depict number plates. They are recorded as `REJECTED_WRONG_CROP` in `LEGACY_PLATE_CROP_AUDIT.csv`; they must never be used for OCR metrics.
4. **Correct crop source regenerated.** Class-3 YOLO boxes yield 24 source-derived plate crops in `outputs/review/label_plate_crops/`, with coordinates in `LABEL_PLATE_CROP_MANIFEST.csv`.
5. **Conservative OCR ledger created.** `PLATE_GROUND_TRUTH_AI_ASSISTED.csv` contains one visually legible transcription (`AP39CV2294`) and 23 `UNREADABLE` decisions. No OCR guess is promoted to ground truth.
6. **Empty-frame review recorded.** Six designated empty frames are recorded in `EMPTY_FRAME_AI_ASSISTED_REVIEW.csv` as `NO_TARGETS_PRESENT` under AI-assisted visual review provenance.

## What cannot be safely automated

The repaired labels still need semantic review. A label file can be geometrically valid while assigning the wrong rider, conflicting helmet classes, or a plate from another motorcycle. Rider--plate ownership cannot be demonstrated reliably by proximity alone. Therefore `FRAME_SEMANTIC_REVIEW_QUEUE.csv` deliberately keeps all 40 frames pending independent semantic review.

## Practical completion method

Use CVAT or Label Studio with the 40 original images and the corrected YOLO labels imported as **pre-annotations**. Configure the four classes exactly as in `data/dataset.yaml`; use one reviewer to amend boxes and a second reviewer to adjudicate every disagreement. The reviewer must use the source image/video, not the detector visualization. For every plate, select exact text only if all characters are legible; otherwise select `UNREADABLE`. Export the reviewed labels, associations, and transcription table with reviewer IDs and timestamps.

For a small corpus, the fastest defensible approach is two-pass manual review rather than retraining. Once all rows in `FRAME_SEMANTIC_REVIEW_QUEUE.csv` have a second-review sign-off, create blocked temporal train/validation/test partitions. The current single-video corpus can support only a within-video test; obtain a separate recording for a generalisation test.

## Training and reporting gate

Do not train/evaluate against old results or report detector/OCR accuracy until the semantic review queue is complete. Runtime results may remain separate because they do not depend on labels. The manuscript must state the AI-assisted provenance and distinguish it from independent human ground truth.
