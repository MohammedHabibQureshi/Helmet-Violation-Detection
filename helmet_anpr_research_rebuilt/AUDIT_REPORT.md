# Helmet Violation Detection — Audit Report
Generated: 2026-10-14T05:04:20.891554Z

## Task 1: Verified On-Disk State
- `results/segment_inference`: EXISTS — 4 file(s): ['frame_0000.jpg', 'frame_0010.jpg', 'frame_0020.jpg', 'inference_summary.json']
- `proposal/`: EXISTS — 0 file(s): []
- `data/`: EXISTS — 6 file(s): ['annotations', 'associations', 'dataset.yaml', 'images', 'labels', 'splits']
- `data/annotations/`: EXISTS — 0 file(s): []
- `data/associations/`: EXISTS — 0 file(s): []
- `proposal/draft_predictions.json`: MISSING

**Video file:** NOT PRESENT on disk. Expected path from config: `D:/hp/Documents/croped-main.mp4` — MISSING.
Source frames used: 3 existing frames from `results/segment_inference/`.

## Task 2: Annotation Review Workflow
- Frames collected: **40**
- Original frames saved to: `data/images/`
- Visualization images saved to: `outputs/review/viz/` (40 files)
- Predictions file: `proposal/draft_predictions.json`
- Review guide: `proposal/annotation_review_guide.md`
- All predictions status: **PROPOSED** (not verified)

## Task 3: Dataset Directory Structure
```
helmet_anpr_research_rebuilt/
  data/
    images/       ← source frames (frame_NNNNNNNN.jpg)
    labels/       ← YOLO .txt files (same stem as image)
    annotations/  ← reserved for CVAT/LabelImg exports
    associations/ ← rider↔plate JSON (same stem + _associations.json)
    splits/       ← train.txt / val.txt / test.txt (fill after labelling)
    dataset.yaml  ← YOLO training config
  outputs/review/viz/  ← PROPOSED visualizations
  proposal/
    draft_predictions.json
    annotation_review_guide.md
```
Image ↔ label mapping: `data/images/frame_00000000.jpg` → `data/labels/frame_00000000.txt`

## Task 4: Model / Checkpoint
- **Weights used:** `Pretrained fallback: D:\hp\Documents\Claude-workspace\Helmet-Violation-Detection\yolov8s.pt`
- **Model variant:** YOLOv8s (pretrained on COCO, 80 classes)
- **Custom 4-class model:** NOT TRAINED — `weights/best.pt` does not exist
- **COCO classes detected ≠ project classes.** The human annotator must
  map each COCO detection to one of: 0=rider, 1=with_helmet,
  2=without_helmet, 3=number_plate — or mark it as irrelevant.

## Task 5: Predictions Summary
| Metric | Value |
|---|---|
| Frames processed | 40 |
| Frames with ≥1 detection | 40 |
| Total PROPOSED detections | 535 |
| Confidence threshold | 0.25 |
| IOU threshold | 0.45 |

### Per-frame breakdown
| Frame | Detections | Classes detected (COCO) |
|---|---|---|
| `frame_00000000.jpg` | 13 | car, motorcycle, person |
| `frame_00000485.jpg` | 13 | car, motorcycle, person, potted plant |
| `frame_00000970.jpg` | 15 | car, motorcycle, person, potted plant |
| `frame_00001455.jpg` | 15 | car, motorcycle, person |
| `frame_00001940.jpg` | 13 | car, motorcycle, person |
| `frame_00002425.jpg` | 16 | car, motorcycle, person, potted plant |
| `frame_00002910.jpg` | 12 | car, motorcycle, person, potted plant |
| `frame_00003395.jpg` | 13 | car, motorcycle, person |
| `frame_00003880.jpg` | 14 | car, motorcycle, person, potted plant |
| `frame_00004365.jpg` | 14 | car, motorcycle, person, potted plant |
| `frame_00004850.jpg` | 15 | car, motorcycle, person, potted plant |
| `frame_00005335.jpg` | 14 | car, motorcycle, potted plant |
| `frame_00005820.jpg` | 15 | backpack, car, motorcycle, person |
| `frame_00006305.jpg` | 16 | car, motorcycle, person, potted plant |
| `frame_00006790.jpg` | 18 | car, motorcycle, person, potted plant |
| `frame_00007275.jpg` | 14 | car, motorcycle, person |
| `frame_00007760.jpg` | 20 | car, motorcycle, person, potted plant |
| `frame_00008245.jpg` | 16 | car, motorcycle, person, potted plant |
| `frame_00008730.jpg` | 14 | car, motorcycle, person, potted plant |
| `frame_00009215.jpg` | 11 | car, motorcycle, person, potted plant |
| `frame_00009700.jpg` | 8 | car, motorcycle, person |
| `frame_00010185.jpg` | 9 | car, motorcycle, person |
| `frame_00010670.jpg` | 10 | car, motorcycle, person |
| `frame_00011155.jpg` | 14 | car, motorcycle, person |
| `frame_00011640.jpg` | 14 | car, motorcycle, person, potted plant |
| `frame_00012125.jpg` | 12 | car, motorcycle, person, potted plant |
| `frame_00012610.jpg` | 12 | car, motorcycle, potted plant |
| `frame_00013095.jpg` | 15 | car, motorcycle, person |
| `frame_00013580.jpg` | 14 | backpack, car, motorcycle, person, potted plant |
| `frame_00014065.jpg` | 13 | car, motorcycle, person, potted plant |
| `frame_00014550.jpg` | 15 | car, motorcycle, person, potted plant |
| `frame_00015035.jpg` | 15 | backpack, car, motorcycle, person, potted plant |
| `frame_00015520.jpg` | 11 | backpack, car, motorcycle, person, potted plant |
| `frame_00016005.jpg` | 11 | car, motorcycle, person |
| `frame_00016490.jpg` | 11 | car, motorcycle, person, potted plant |
| `frame_00016975.jpg` | 12 | car, motorcycle, person, potted plant |
| `frame_00017460.jpg` | 12 | car, motorcycle, person, potted plant |
| `frame_00017945.jpg` | 13 | car, motorcycle, person, potted plant |
| `frame_00018430.jpg` | 13 | car, motorcycle, person, potted plant |
| `frame_00018915.jpg` | 10 | car, motorcycle, person |

## Remaining Human Actions Required
1. **Provide the video file** at `D:/hp/Documents/croped-main.mp4` so more
   diverse frames can be extracted. Re-run this script afterward.
2. **Open review images** and verify PROPOSED boxes:
   ```cmd
   start D:\hp\Documents\Claude-workspace\Helmet-Violation-Detection\helmet_anpr_research_rebuilt\outputs\review\viz
   ```
3. **Map COCO classes → project classes** in `proposal/draft_predictions.json`.
   Set `project_class_id` and change `status` from `PROPOSED` to `VERIFIED`.
4. **Write YOLO label files** in `data/labels/*.txt` (one line per object).
5. **Write association files** in `data/associations/*_associations.json`.
6. **Run annotation validator** to catch coordinate errors:
   ```cmd
   python -I helmet_anpr_research_rebuilt/src/annotation_validator.py
   ```
7. **Populate `data/splits/`** with train.txt / val.txt / test.txt.
8. **Train custom model** after annotations are verified (see review guide).

## Paths Created by This Script
- `helmet_anpr_research_rebuilt\data\images`
- `helmet_anpr_research_rebuilt\data\labels`
- `helmet_anpr_research_rebuilt\data\annotations`
- `helmet_anpr_research_rebuilt\data\associations`
- `helmet_anpr_research_rebuilt\proposal`
- `helmet_anpr_research_rebuilt\outputs\review\viz`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00000000.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00000485.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00000970.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00001455.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00001940.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00002425.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00002910.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00003395.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00003880.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00004365.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00004850.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00005335.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00005820.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00006305.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00006790.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00007275.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00007760.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00008245.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00008730.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00009215.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00009700.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00010185.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00010670.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00011155.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00011640.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00012125.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00012610.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00013095.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00013580.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00014065.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00014550.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00015035.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00015520.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00016005.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00016490.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00016975.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00017460.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00017945.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00018430.jpg`
- `helmet_anpr_research_rebuilt\outputs\review\viz\frame_00018915.jpg`
- `helmet_anpr_research_rebuilt\proposal\draft_predictions.json`
- `helmet_anpr_research_rebuilt\proposal\annotation_review_guide.md`
- `helmet_anpr_research_rebuilt\data\labels\frame_00000000.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00000485.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00000970.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00001455.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00001940.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00002425.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00002910.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00003395.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00003880.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00004365.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00004850.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00005335.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00005820.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00006305.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00006790.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00007275.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00007760.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00008245.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00008730.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00009215.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00009700.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00010185.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00010670.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00011155.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00011640.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00012125.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00012610.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00013095.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00013580.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00014065.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00014550.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00015035.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00015520.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00016005.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00016490.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00016975.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00017460.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00017945.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00018430.txt`
- `helmet_anpr_research_rebuilt\data\labels\frame_00018915.txt`
- `helmet_anpr_research_rebuilt\data\associations\frame_00000000_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00000485_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00000970_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00001455_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00001940_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00002425_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00002910_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00003395_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00003880_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00004365_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00004850_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00005335_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00005820_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00006305_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00006790_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00007275_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00007760_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00008245_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00008730_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00009215_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00009700_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00010185_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00010670_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00011155_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00011640_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00012125_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00012610_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00013095_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00013580_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00014065_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00014550_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00015035_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00015520_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00016005_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00016490_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00016975_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00017460_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00017945_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00018430_associations.json`
- `helmet_anpr_research_rebuilt\data\associations\frame_00018915_associations.json`
