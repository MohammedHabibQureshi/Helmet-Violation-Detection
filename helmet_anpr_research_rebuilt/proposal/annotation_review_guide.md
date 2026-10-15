# Annotation Review Guide
Generated: 2026-10-14T05:04:20.890200Z  
Model used: `Pretrained fallback: D:\hp\Documents\Claude-workspace\Helmet-Violation-Detection\yolov8s.pt`  
**All predictions are PROPOSED. Do not use them as ground truth without verification.**

## Class Mapping
| ID | Name |
|---|---|
| 0 | `rider` |
| 1 | `with_helmet` |
| 2 | `without_helmet` |
| 3 | `number_plate` |

## Directory Layout
```
helmet_anpr_research_rebuilt/
  data/images/          ← original frames (read-only source)
  outputs/review/viz/   ← visualizations with PROPOSED boxes
  proposal/             ← draft_predictions.json (PROPOSED)
  data/labels/          ← YOLO .txt stubs (EMPTY — fill these)
  data/associations/    ← rider↔plate association stubs
```

## How to Open Review Images (Windows)
```cmd
# Open all viz images in Windows Photo Viewer:
start D:\hp\Documents\Claude-workspace\Helmet-Violation-Detection\helmet_anpr_research_rebuilt\outputs\review\viz
```

## Annotation Workflow
1. Open each image in `outputs/review/viz/` to see PROPOSED boxes.
2. Open `proposal/draft_predictions.json` — each detection has:
   - `coco_class_name`: what the pretrained model detected
   - `project_class_id` / `project_class_name`: **set these to the correct class**
   - `status`: change from `PROPOSED` to `VERIFIED` when correct
3. Edit `data/labels/<frame_stem>.txt` with verified YOLO boxes:
   ```
   <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
   ```
   Coordinates must be **normalized** (0–1) relative to image size.
4. Edit `data/associations/<frame_stem>_associations.json` to link
   each helmetless rider detection to its number plate detection.
5. Run the validator:
   ```cmd
   python -I helmet_anpr_research_rebuilt/src/annotation_validator.py
   ```

## YOLO Label Format Reference
```
# One row per object:
# <class_id> <cx> <cy> <w> <h>   (all normalized 0-1)
#
# Example: rider occupying left half of a 1920x1080 image
0 0.25 0.50 0.50 1.00
# number_plate in bottom-right corner:
3 0.85 0.90 0.10 0.05
```

## Predicted Frames Summary
| Frame | Detections | Inference ms |
|---|---|---|
| `frame_00000000.jpg` | 13 | 3274.879 |
| `frame_00000485.jpg` | 13 | 13.728 |
| `frame_00000970.jpg` | 15 | 18.454 |
| `frame_00001455.jpg` | 15 | 14.856 |
| `frame_00001940.jpg` | 13 | 13.193 |
| `frame_00002425.jpg` | 16 | 12.445 |
| `frame_00002910.jpg` | 12 | 13.513 |
| `frame_00003395.jpg` | 13 | 14.542 |
| `frame_00003880.jpg` | 14 | 33.3 |
| `frame_00004365.jpg` | 14 | 12.769 |
| `frame_00004850.jpg` | 15 | 12.84 |
| `frame_00005335.jpg` | 14 | 12.439 |
| `frame_00005820.jpg` | 15 | 13.949 |
| `frame_00006305.jpg` | 16 | 12.794 |
| `frame_00006790.jpg` | 18 | 13.047 |
| `frame_00007275.jpg` | 14 | 12.244 |
| `frame_00007760.jpg` | 20 | 13.152 |
| `frame_00008245.jpg` | 16 | 12.371 |
| `frame_00008730.jpg` | 14 | 12.99 |
| `frame_00009215.jpg` | 11 | 12.999 |
| `frame_00009700.jpg` | 8 | 12.637 |
| `frame_00010185.jpg` | 9 | 12.759 |
| `frame_00010670.jpg` | 10 | 13.039 |
| `frame_00011155.jpg` | 14 | 12.31 |
| `frame_00011640.jpg` | 14 | 12.504 |
| `frame_00012125.jpg` | 12 | 12.29 |
| `frame_00012610.jpg` | 12 | 12.38 |
| `frame_00013095.jpg` | 15 | 12.228 |
| `frame_00013580.jpg` | 14 | 12.226 |
| `frame_00014065.jpg` | 13 | 14.211 |
| `frame_00014550.jpg` | 15 | 12.706 |
| `frame_00015035.jpg` | 15 | 12.782 |
| `frame_00015520.jpg` | 11 | 12.566 |
| `frame_00016005.jpg` | 11 | 13.526 |
| `frame_00016490.jpg` | 11 | 12.41 |
| `frame_00016975.jpg` | 12 | 12.368 |
| `frame_00017460.jpg` | 12 | 12.433 |
| `frame_00017945.jpg` | 13 | 12.431 |
| `frame_00018430.jpg` | 13 | 13.441 |
| `frame_00018915.jpg` | 10 | 12.406 |

## Validation Command (after labelling)
```cmd
cd /d D:\hp\Documents\Claude-workspace\Helmet-Violation-Detection
python -I helmet_anpr_research_rebuilt/src/annotation_validator.py
```

## Training Command (after annotations verified)
```cmd
cd /d D:\hp\Documents\Claude-workspace\Helmet-Violation-Detection
python -I -c "
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.train(data='helmet_anpr_research_rebuilt/data/dataset.yaml',
            epochs=100, imgsz=640, batch=16, device=0,
            project='helmet_anpr_research_rebuilt/results/training',
            name='helmet_yolov8s')
"
```
