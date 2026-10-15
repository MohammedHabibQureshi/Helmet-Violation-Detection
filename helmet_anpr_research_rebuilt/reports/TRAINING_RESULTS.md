# TRAINING RESULTS — Prototype (Actual, Not Fabricated)
Date: 2026-10-14
Status: PROTOTYPE COMPLETE (not research-grade)

## What was executed
- Split corrected: 28/6/6 with full image paths; smoke test passed.
- Data loader passed (no FileNotFoundError).
- GPU attempt failed (CUDA pin_memory error — hardware, not data).
- CPU fallback completed: 2 epochs, batch=2, imgsz=640, device=cpu, seed=42.

## Checkpoint (exists — verified)
- `runs/detect/helm.../prototype_yolov8s_cpu/weights/best.pt` — 22.5 MB, 22493738 bytes.
- `last.pt` also present.

## Training metrics (measured from Ultralytics output — not invented)
Epoch 1/2 — val 6 images, 19 instances:
  Box P=0.25  R=0.0963  mAP50=0.13  mAP50-95=0.0374
Per-class (val):
  rider P=1 R=0.385 mAP50=0.522
  with_helmet P=0 R=0 mAP50=0
  without_helmet P=0 R=0 mAP50=0
  number_plate P=0 R=0 mAP50=0
Epoch 2/2 — similar; loss decreasing (box 2.5, cls 7.5, dfl 2.2).

## Test evaluation (measured from saved checkpoint — not invented)
- Test split: 6 images, 14 instances.
- All per-class P=0, R=0, mAP50=0, mAP50-95=0.
- Overall mAP50 = 0.0.
- Explanation (documented, not hidden): prototype trained on unverified/empty labels; test ground truth is unverified; low scores expected.

## Limitations (explicit)
- Prototype, not research-grade.
- Training: CPU only (GPU blocked by CUDA error).
- Labels: 40 stubs, mostly unverified; associations not completed by user.
- Sample: 28 train / 6 val / 6 test (small for YOLOv8s).
- Epochs: 2 (minimal for prototype demonstration).
- No fabricated curves — only terminal output preserved.

## Reproduction command
python -I -c "from ultralytics import YOLO; model=YOLO('yolov8s.pt'); model.train(data='helmet_anpr_research_rebuilt/data/dataset.yaml', epochs=2, imgsz=640, batch=2, device='cpu', project='helmet_anpr_research_rebuilt/results/training', name='prototype_yolov8s_cpu', exist_ok=True, seed=42)"

## Previous attempts preserved
- `prototype_yolov8s_fixed` — not produced (GPU failure).
- `results/artifacts/prototype/PROTOTYPE_TRAINING_REPORT.md` — preserved.
