# Verification release gate -- 2026-10-14

## Decision

**DO NOT FREEZE, TRAIN, OR REPORT ACCURACY FROM THE CURRENT LABELS.**

The audit confirms 40 image/label pairs and 101 syntactically valid YOLO annotations. It does not establish semantic ground truth. All rider-to-plate records are `AUTO_PROPOSED`; no independently reviewed transcription file exists. Changing their status to `VERIFIED` would be a false provenance claim.

## Evidence reviewed

- Source video: `croped-main.mp4`; 40 sampled images are retained in `data/images/`.
- Labels: 40 files, 101 boxes; structural validation passed without coordinate or class errors.
- Associations: 27 proposed links; 0 confirmed links.
- Plate crops: 32 retained review crops; 0 approved transcriptions.
- Empty-label candidates: `frame_00006790`, `frame_00012610`, `frame_00014065`, `frame_00015035`, `frame_00016490`, and `frame_00017945`. These require source-video adjudication; they are not true negatives yet.

## Split finding

The manifests put all 40 images in `train.txt` and leave validation/test empty. Because material is one continuous video, it cannot support an independent generalisation test. After review, create non-overlapping temporal blocks with a boundary buffer and reserve a separate recording/camera for final generalisation testing.

## Required sign-off

For every frame retain reviewer, timestamp, native-resolution evidence, box-level disposition, and a second-review outcome. For every crop record a literal transcription, `UNREADABLE`, or `NOT_A_PLATE`. For every association record `CONFIRMED`, `REJECTED`, or `UNRESOLVABLE` plus linked box IDs.

## Metrics gate

Do not calculate detection AP/F1/mAP, OCR CER/exact match, association accuracy, or select a training checkpoint until these records are signed and the held-out evaluation partition is frozen. Runtime measurements can be reported separately with their exact segment conditions.
