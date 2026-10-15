# Annotation Tool Audit — `annotate.py` v2

Date: 2026-10-14

## Honest audit of the original `annotate.py` (v1)

The v1 tool only supported *assigning a project class to an existing COCO box*.
It did **not** support drawing, moving, resizing, or deleting boxes. It also had
a bug: reopening a frame reloaded the COCO proposals, so REJECTED and UNCERTAIN
decisions from a previous session were silently lost.

| Requirement | v1 status |
|---|---|
| Draw new boxes | **MISSING** |
| Move boxes | **MISSING** |
| Resize boxes | **MISSING** |
| Delete boxes | **MISSING** (docstring said "not implemented") |
| Rider box separate from motorcycle box | Partially — only if COCO detected a `person` |
| Helmet status only after visual verification | OK |
| Manual plate box when COCO missed it | **MISSING** |
| Valid YOLO format save | OK |
| Association JSON schema | OK |
| REJECTED/UNCERTAIN excluded from labels | OK |
| SPACE does not silently verify unreviewed boxes | OK |
| Reopen preserves all decisions | **BUG** — REJECTED/UNCERTAIN lost on reopen |

## `annotate.py` v2 — what changed

Rewrote the entire tool with a full mouse state machine and per-frame decision
persistence. Backward-compatible with existing label files.

### New capabilities

| Feature | How it works |
|---|---|
| **Draw new box** | Left-click + drag anywhere on the image. New box starts as PROPOSED, then assign a class with keys 1-4. |
| **Move box** | Middle-click inside the selected box, then drag. |
| **Resize box** | Right-click on a corner handle (white square), then drag. Nearest corner is auto-selected. |
| **Delete box** | Press `Delete` or `D`. Removes the selected box entirely. |
| **Reopen preserves decisions** | Every frame's decisions (VERIFIED/REJECTED/UNCERTAIN) are saved to `proposal/decisions/<stem>_decisions.json`. Reopening a frame reloads them — no duplication, no reset. |

### Visual distinction

| Style | Meaning |
|---|---|
| Grey dashed | PROPOSED — unreviewed (COCO detection or new manual draw) |
| Green solid | VERIFIED — human-assigned project class |
| Red solid | REJECTED — confirmed false positive |
| Yellow solid | UNCERTAIN — ambiguous or unreadable |
| Cyan thick outline | Currently selected box |
| White squares on selected box | Resize handles (right-click to drag) |

## Test results — `test_annotate_logic.py`

**38 checks passed, 0 failed.**

### Requirement-by-requirement

| Req | Description | Result |
|---|---|---|
| 1 | Rider box separate from motorcycle box | **PASS** — reject the COCO motorcycle box, then draw a new rider box manually; they are independent objects |
| 2 | Helmet status only after explicit keypress | **PASS** — box starts PROPOSED; only a keypress sets VERIFIED |
| 3 | Manual plate box when COCO missed it | **PASS** — LMB drag draws a box even with zero COCO detections; assigned class 3 with plate text |
| 4 | Valid YOLO format save | **PASS** — 5 fields, class_id correct, all coords in [0,1], cx/cy/w/h normalized |
| 4b | Association JSON schema | **PASS** — rider_box_idx, plate_box_idx, has_helmet (bool), plate_text (str), status=VERIFIED |
| 5 | REJECTED/UNCERTAIN excluded from labels | **PASS** — only VERIFIED boxes written; REJECTED, UNCERTAIN, and unreviewed PROPOSED all absent |
| 6 | SPACE does not silently verify unreviewed boxes | **PASS** — PROPOSED box stays PROPOSED after save; not written to label file |
| 7 | Reopen preserves all decisions | **PASS** — VERIFIED stays VERIFIED, REJECTED stays REJECTED, PROPOSED stays PROPOSED; no duplication, no reset |
| — | Move box (MMB drag) | **PASS** — box position changes |
| — | Resize box (RMB drag on corner) | **PASS** — box dimensions change |
| — | Delete box | **PASS** — correct box removed |

### Validator test results — `test_validator.py`

**14 checks passed, 0 failed.** Covers valid inputs (single box, all four classes,
comment/blank line skipping, empty file, plate at minimum size), invalid inputs
(invalid class ID, out-of-range coords, negative coords, below-minimum width,
wrong field count, non-numeric values, duplicate boxes, bad plate aspect ratio,
multiple errors in one file), and dataset-level orphan detection.

## Confirmed annotation command

```cmd
cd /d D:\hp\Documents\Claude-workspace\Helmet-Violation-Detection
python -I annotate.py
```

Start from a specific frame (0-based):
```cmd
python -I annotate.py --start 10
```

## Files to review

| File | Purpose |
|---|---|
| `outputs/review/viz/*.jpg` (40 files) | Visual inspection of PROPOSED boxes |
| `proposal/draft_predictions.json` | Raw COCO predictions (PROPOSED) |
| `proposal/annotation_review_guide.md` | Full annotation reference |
| `proposal/decisions/*.json` (created on first save) | Per-frame decision persistence |
| `data/labels/*.txt` (40 empty stubs) | Fill via annotator |
| `data/associations/*_associations.json` (40 stubs) | Fill via annotator |

## Status

- Custom YOLO training: **NOT STARTED** — waiting for verified annotations
- No annotations, plate characters, associations, or accuracy metrics have been invented