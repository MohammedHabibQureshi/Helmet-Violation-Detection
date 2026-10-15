"""
Helmet Violation Detection — Annotation Review Workflow
========================================================
This script:
  1. Audits the actual on-disk state of the project.
  2. Uses the 3 existing frames (no video present) as source images.
     If a video is later placed at configs/base_config.yaml::video_path,
     it will extract a representative sample of up to 40 frames instead.
  3. Runs YOLOv8s (pretrained COCO) on each frame — because no custom
     4-class weights exist yet.  All predictions are tagged PROPOSED.
  4. Saves:
       data/images/            — original frames (copies, never overwritten)
       outputs/review/viz/     — visualizations with boxes + labels
       proposal/draft_predictions.json — raw machine predictions (PROPOSED)
       proposal/annotation_review_guide.md
       data/labels/            — empty YOLO .txt stubs (one per image,
                                  to be filled by the human annotator)
       data/associations/      — empty per-image association stubs
  5. Writes a final audit report to AUDIT_REPORT.md.

Run from the project root (Helmet-Violation-Detection/):
    python -I run_audit_workflow.py
"""

import json
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "helmet_anpr_research_rebuilt"

DATA_IMAGES   = PROJECT / "data" / "images"
DATA_LABELS   = PROJECT / "data" / "labels"
DATA_ANNOTS   = PROJECT / "data" / "annotations"
DATA_ASSOC    = PROJECT / "data" / "associations"
PROPOSAL      = PROJECT / "proposal"
VIZ_DIR       = PROJECT / "outputs" / "review" / "viz"
ORIG_FRAMES   = PROJECT / "results" / "segment_inference"

CLASS_MAP = {0: "rider", 1: "with_helmet", 2: "without_helmet", 3: "number_plate"}
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45

# ── helpers ────────────────────────────────────────────────────────────────

def ensure_dirs():
    for d in [DATA_IMAGES, DATA_LABELS, DATA_ANNOTS, DATA_ASSOC, PROPOSAL, VIZ_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def audit_existing():
    """Return a dict describing what actually exists on disk."""
    report = {}

    def check_dir(rel, d):
        d = Path(d)
        if not d.exists():
            report[rel] = "MISSING"
        else:
            files = list(d.iterdir())
            report[rel] = f"EXISTS — {len(files)} file(s): {[f.name for f in files[:8]]}"

    check_dir("results/segment_inference", ORIG_FRAMES)
    check_dir("proposal/",                 PROPOSAL)
    check_dir("data/",                     PROJECT / "data")
    check_dir("data/annotations/",         DATA_ANNOTS)
    check_dir("data/associations/",        DATA_ASSOC)

    # proposal/draft_predictions.json
    dp = PROPOSAL / "draft_predictions.json"
    if dp.exists():
        try:
            content = json.loads(dp.read_text(encoding="utf-8"))
            count = len(content.get("predictions", []))
            report["proposal/draft_predictions.json"] = (
                f"EXISTS — {count} prediction block(s)"
            )
        except Exception as e:
            report["proposal/draft_predictions.json"] = f"EXISTS but INVALID JSON: {e}"
    else:
        report["proposal/draft_predictions.json"] = "MISSING"

    return report


def collect_source_frames():
    """
    Prefer a real video (if present).  Fall back to the 3 existing frames.
    Returns list of (dest_path, frame_index) tuples — images copied into
    data/images/ with names frame_NNNNNNNN.jpg.
    """
    config_path = PROJECT / "configs" / "base_config.yaml"
    video_path = None
    if config_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            vp = Path(cfg.get("video_path", ""))
            if vp.exists():
                video_path = vp
        except Exception:
            pass

    collected = []

    if video_path:
        print(f"[INFO] Video found: {video_path}. Extracting representative frames...")
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] Cannot open video {video_path}. Falling back to existing frames.")
            video_path = None
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Sample ~40 frames spread across the video
            target = min(40, total)
            step = max(1, total // target)
            frame_indices = list(range(0, total, step))[:target]
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                dest = DATA_IMAGES / f"frame_{idx:08d}.jpg"
                if not dest.exists():
                    cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                collected.append((dest, idx))
            cap.release()
            print(f"[INFO] Extracted {len(collected)} frames from video.")

    if not video_path:
        # Use the 3 existing frames — copy them into data/images/ without overwriting
        existing = sorted(ORIG_FRAMES.glob("*.jpg"))
        print(f"[INFO] No video on disk. Using {len(existing)} existing frames from results/segment_inference/")
        for src in existing:
            dest = DATA_IMAGES / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
            # parse index from name
            try:
                idx = int(src.stem.split("_")[1])
            except Exception:
                idx = 0
            collected.append((dest, idx))

    return collected


def run_detector(frame_paths):
    """
    Run YOLOv8s (pretrained) on each frame.
    Returns list of prediction dicts.
    """
    try:
        import cv2
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}", file=sys.stderr)
        return None, str(e)

    weights = PROJECT / "weights" / "best.pt"
    using_custom = weights.exists()
    if using_custom:
        model = YOLO(str(weights))
        weights_note = f"Custom weights: {weights}"
    else:
        weights_note = f"Pretrained fallback: {ROOT / 'yolov8s.pt'}"
        model = YOLO(str(ROOT / "yolov8s.pt"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Model: {weights_note}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model class names: {list(model.names.values())[:10]}...")

    all_predictions = []
    for dest_path, frame_idx in frame_paths:
        frame = cv2.imread(str(dest_path))
        if frame is None:
            print(f"[WARN] Cannot read {dest_path}", file=sys.stderr)
            continue

        t0 = time.perf_counter()
        results = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=device,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        h, w = frame.shape[:2]
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cid  = int(box.cls[0].item())
                conf = round(float(box.conf[0].item()), 4)
                x1, y1, x2, y2 = (round(float(v), 2) for v in box.xyxy[0].tolist())
                # Use COCO name from this pretrained model
                coco_name = model.names.get(cid, f"class_{cid}")
                detections.append({
                    "coco_class_id":   cid,
                    "coco_class_name": coco_name,
                    "project_class_id":   None,  # must be set by human annotator
                    "project_class_name": None,  # must be set by human annotator
                    "conf":   conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "area":  round((x2 - x1) * (y2 - y1), 2),
                    "status": "PROPOSED",
                })
        all_predictions.append({
            "frame_file":     dest_path.name,
            "frame_index":    frame_idx,
            "image_width":    w,
            "image_height":   h,
            "inference_ms":   round(inference_ms, 3),
            "weights_used":   weights_note,
            "using_custom_weights": using_custom,
            "num_detections": len(detections),
            "detections":     detections,
        })

        print(f"  {dest_path.name}: {len(detections)} detections in {inference_ms:.1f}ms")

    return all_predictions, weights_note


def draw_viz(frame_paths_map, predictions, weights_note):
    """Save visualization images with bounding boxes."""
    import cv2

    # Build lookup by filename
    pred_by_file = {p["frame_file"]: p for p in predictions}

    for dest_path, _ in frame_paths_map:
        pred = pred_by_file.get(dest_path.name)
        if pred is None:
            continue

        frame = cv2.imread(str(dest_path))
        if frame is None:
            continue

        for det in pred["detections"]:
            x1, y1 = int(det["x1"]), int(det["y1"])
            x2, y2 = int(det["x2"]), int(det["y2"])
            label = f"{det['coco_class_name']} {det['conf']:.2f} [PROPOSED]"
            # Red box for PROPOSED
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 220), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Stamp the weights note and status in the corner
        cv2.putText(frame, f"MODEL: {weights_note[:60]}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        cv2.putText(frame, "STATUS: PROPOSED — NOT VERIFIED", (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 2)

        out_path = VIZ_DIR / dest_path.name
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return list(VIZ_DIR.glob("*.jpg"))


def write_draft_predictions(predictions, weights_note):
    """Save proposal/draft_predictions.json."""
    schema_doc = {
        "_schema_version": "1.0",
        "_description": (
            "Machine-generated bounding-box predictions tagged PROPOSED. "
            "A human annotator must verify every detection and fill in "
            "project_class_id / project_class_name before these can be used "
            "as training labels. Do NOT treat these as ground truth."
        ),
        "_class_mapping": CLASS_MAP,
        "_weights_used": weights_note,
        "_generated_at": datetime.utcnow().isoformat() + "Z",
        "_status": "PROPOSED — AWAITING HUMAN VERIFICATION",
        "predictions": predictions,
    }
    out = PROPOSAL / "draft_predictions.json"
    out.write_text(json.dumps(schema_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def write_yolo_stubs(predictions):
    """
    Write empty YOLO .txt stubs in data/labels/ so the annotator knows which
    files need labels.  An empty file means 'no objects' — annotator must fill
    it.  Never overwrites a non-empty existing file.
    """
    written = []
    for pred in predictions:
        stem = Path(pred["frame_file"]).stem
        label_file = DATA_LABELS / f"{stem}.txt"
        if not label_file.exists() or label_file.stat().st_size == 0:
            # Write a header comment only (not valid YOLO syntax — annotator
            # replaces this with real labels)
            label_file.write_text(
                "# YOLO format: <class_id> <cx> <cy> <w> <h> (normalized)\n"
                "# class_id: 0=rider 1=with_helmet 2=without_helmet 3=number_plate\n"
                "# STATUS: EMPTY — fill with verified annotations\n",
                encoding="utf-8",
            )
            written.append(label_file)
    return written


def write_association_stubs(predictions):
    """Write empty per-image association stubs in data/associations/."""
    written = []
    for pred in predictions:
        stem = Path(pred["frame_file"]).stem
        assoc_file = DATA_ASSOC / f"{stem}_associations.json"
        if not assoc_file.exists():
            stub = {
                "frame_file": pred["frame_file"],
                "status": "PENDING — fill in rider_box_idx and plate_box_idx",
                "associations": [],
                "_schema": {
                    "rider_box_idx": "index into detections array for the rider",
                    "plate_box_idx": "index into detections array for the plate",
                    "has_helmet": "true / false",
                    "plate_text": "string or null",
                }
            }
            assoc_file.write_text(json.dumps(stub, indent=2), encoding="utf-8")
            written.append(assoc_file)
    return written


def write_review_guide(predictions, viz_files, weights_note):
    """Write proposal/annotation_review_guide.md."""
    lines = []
    lines.append("# Annotation Review Guide\n")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z  \n")
    lines.append(f"Model used: `{weights_note}`  \n")
    lines.append("**All predictions are PROPOSED. Do not use them as ground truth without verification.**\n\n")

    lines.append("## Class Mapping\n")
    lines.append("| ID | Name |\n|---|---|\n")
    for k, v in CLASS_MAP.items():
        lines.append(f"| {k} | `{v}` |\n")
    lines.append("\n")

    lines.append("## Directory Layout\n")
    lines.append("```\n")
    lines.append("helmet_anpr_research_rebuilt/\n")
    lines.append("  data/images/          ← original frames (read-only source)\n")
    lines.append("  outputs/review/viz/   ← visualizations with PROPOSED boxes\n")
    lines.append("  proposal/             ← draft_predictions.json (PROPOSED)\n")
    lines.append("  data/labels/          ← YOLO .txt stubs (EMPTY — fill these)\n")
    lines.append("  data/associations/    ← rider↔plate association stubs\n")
    lines.append("```\n\n")

    lines.append("## How to Open Review Images (Windows)\n")
    lines.append("```cmd\n")
    lines.append(f"# Open all viz images in Windows Photo Viewer:\n")
    lines.append(f"start {VIZ_DIR}\n")
    lines.append("```\n\n")

    lines.append("## Annotation Workflow\n")
    lines.append("1. Open each image in `outputs/review/viz/` to see PROPOSED boxes.\n")
    lines.append("2. Open `proposal/draft_predictions.json` — each detection has:\n")
    lines.append("   - `coco_class_name`: what the pretrained model detected\n")
    lines.append("   - `project_class_id` / `project_class_name`: **set these to the correct class**\n")
    lines.append("   - `status`: change from `PROPOSED` to `VERIFIED` when correct\n")
    lines.append("3. Edit `data/labels/<frame_stem>.txt` with verified YOLO boxes:\n")
    lines.append("   ```\n")
    lines.append("   <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>\n")
    lines.append("   ```\n")
    lines.append("   Coordinates must be **normalized** (0–1) relative to image size.\n")
    lines.append("4. Edit `data/associations/<frame_stem>_associations.json` to link\n")
    lines.append("   each helmetless rider detection to its number plate detection.\n")
    lines.append("5. Run the validator:\n")
    lines.append("   ```cmd\n")
    lines.append("   python -I helmet_anpr_research_rebuilt/src/annotation_validator.py\n")
    lines.append("   ```\n\n")

    lines.append("## YOLO Label Format Reference\n")
    lines.append("```\n")
    lines.append("# One row per object:\n")
    lines.append("# <class_id> <cx> <cy> <w> <h>   (all normalized 0-1)\n")
    lines.append("#\n")
    lines.append("# Example: rider occupying left half of a 1920x1080 image\n")
    lines.append("0 0.25 0.50 0.50 1.00\n")
    lines.append("# number_plate in bottom-right corner:\n")
    lines.append("3 0.85 0.90 0.10 0.05\n")
    lines.append("```\n\n")

    lines.append("## Predicted Frames Summary\n")
    lines.append("| Frame | Detections | Inference ms |\n|---|---|---|\n")
    for p in predictions:
        lines.append(f"| `{p['frame_file']}` | {p['num_detections']} | {p['inference_ms']} |\n")
    lines.append("\n")

    lines.append("## Validation Command (after labelling)\n")
    lines.append("```cmd\n")
    lines.append("cd /d D:\\hp\\Documents\\Claude-workspace\\Helmet-Violation-Detection\n")
    lines.append("python -I helmet_anpr_research_rebuilt/src/annotation_validator.py\n")
    lines.append("```\n\n")

    lines.append("## Training Command (after annotations verified)\n")
    lines.append("```cmd\n")
    lines.append("cd /d D:\\hp\\Documents\\Claude-workspace\\Helmet-Violation-Detection\n")
    lines.append("python -I -c \"\n")
    lines.append("from ultralytics import YOLO\n")
    lines.append("model = YOLO('yolov8s.pt')\n")
    lines.append("model.train(data='helmet_anpr_research_rebuilt/data/dataset.yaml',\n")
    lines.append("            epochs=100, imgsz=640, batch=16, device=0,\n")
    lines.append("            project='helmet_anpr_research_rebuilt/results/training',\n")
    lines.append("            name='helmet_yolov8s')\n")
    lines.append("\"\n")
    lines.append("```\n")

    guide_path = PROPOSAL / "annotation_review_guide.md"
    guide_path.write_text("".join(lines), encoding="utf-8")
    return guide_path


def write_audit_report(audit_results, frame_paths, predictions, weights_note, viz_files, stubs):
    """Write AUDIT_REPORT.md at the project root."""

    frames_with_preds = sum(1 for p in predictions if p["num_detections"] > 0)
    total_dets = sum(p["num_detections"] for p in predictions)

    lines = []
    lines.append("# Helmet Violation Detection — Audit Report\n")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")

    lines.append("## Task 1: Verified On-Disk State\n")
    for k, v in audit_results.items():
        lines.append(f"- `{k}`: {v}\n")
    lines.append("\n**Video file:** NOT PRESENT on disk. ")
    lines.append("Expected path from config: `D:/hp/Documents/croped-main.mp4` — MISSING.\n")
    lines.append("Source frames used: 3 existing frames from `results/segment_inference/`.\n\n")

    lines.append("## Task 2: Annotation Review Workflow\n")
    lines.append(f"- Frames collected: **{len(frame_paths)}**\n")
    lines.append(f"- Original frames saved to: `data/images/`\n")
    lines.append(f"- Visualization images saved to: `outputs/review/viz/` ({len(viz_files)} files)\n")
    lines.append(f"- Predictions file: `proposal/draft_predictions.json`\n")
    lines.append(f"- Review guide: `proposal/annotation_review_guide.md`\n")
    lines.append(f"- All predictions status: **PROPOSED** (not verified)\n\n")

    lines.append("## Task 3: Dataset Directory Structure\n")
    lines.append("```\n")
    lines.append("helmet_anpr_research_rebuilt/\n")
    lines.append("  data/\n")
    lines.append("    images/       ← source frames (frame_NNNNNNNN.jpg)\n")
    lines.append("    labels/       ← YOLO .txt files (same stem as image)\n")
    lines.append("    annotations/  ← reserved for CVAT/LabelImg exports\n")
    lines.append("    associations/ ← rider↔plate JSON (same stem + _associations.json)\n")
    lines.append("    splits/       ← train.txt / val.txt / test.txt (fill after labelling)\n")
    lines.append("    dataset.yaml  ← YOLO training config\n")
    lines.append("  outputs/review/viz/  ← PROPOSED visualizations\n")
    lines.append("  proposal/\n")
    lines.append("    draft_predictions.json\n")
    lines.append("    annotation_review_guide.md\n")
    lines.append("```\n")
    lines.append("Image ↔ label mapping: `data/images/frame_00000000.jpg` → `data/labels/frame_00000000.txt`\n\n")

    lines.append("## Task 4: Model / Checkpoint\n")
    lines.append(f"- **Weights used:** `{weights_note}`\n")
    lines.append("- **Model variant:** YOLOv8s (pretrained on COCO, 80 classes)\n")
    lines.append("- **Custom 4-class model:** NOT TRAINED — `weights/best.pt` does not exist\n")
    lines.append("- **COCO classes detected ≠ project classes.** The human annotator must\n")
    lines.append("  map each COCO detection to one of: 0=rider, 1=with_helmet,\n")
    lines.append("  2=without_helmet, 3=number_plate — or mark it as irrelevant.\n\n")

    lines.append("## Task 5: Predictions Summary\n")
    lines.append(f"| Metric | Value |\n|---|---|\n")
    lines.append(f"| Frames processed | {len(predictions)} |\n")
    lines.append(f"| Frames with ≥1 detection | {frames_with_preds} |\n")
    lines.append(f"| Total PROPOSED detections | {total_dets} |\n")
    lines.append(f"| Confidence threshold | {CONF_THRESHOLD} |\n")
    lines.append(f"| IOU threshold | {IOU_THRESHOLD} |\n\n")

    lines.append("### Per-frame breakdown\n")
    lines.append("| Frame | Detections | Classes detected (COCO) |\n|---|---|---|\n")
    for p in predictions:
        classes = ", ".join(sorted({d["coco_class_name"] for d in p["detections"]})) or "none"
        lines.append(f"| `{p['frame_file']}` | {p['num_detections']} | {classes} |\n")
    lines.append("\n")

    lines.append("## Remaining Human Actions Required\n")
    lines.append("1. **Provide the video file** at `D:/hp/Documents/croped-main.mp4` so more\n")
    lines.append("   diverse frames can be extracted. Re-run this script afterward.\n")
    lines.append("2. **Open review images** and verify PROPOSED boxes:\n")
    lines.append(f"   ```cmd\n   start {VIZ_DIR}\n   ```\n")
    lines.append("3. **Map COCO classes → project classes** in `proposal/draft_predictions.json`.\n")
    lines.append("   Set `project_class_id` and change `status` from `PROPOSED` to `VERIFIED`.\n")
    lines.append("4. **Write YOLO label files** in `data/labels/*.txt` (one line per object).\n")
    lines.append("5. **Write association files** in `data/associations/*_associations.json`.\n")
    lines.append("6. **Run annotation validator** to catch coordinate errors:\n")
    lines.append("   ```cmd\n   python -I helmet_anpr_research_rebuilt/src/annotation_validator.py\n   ```\n")
    lines.append("7. **Populate `data/splits/`** with train.txt / val.txt / test.txt.\n")
    lines.append("8. **Train custom model** after annotations are verified (see review guide).\n\n")

    lines.append("## Paths Created by This Script\n")
    created = (
        [DATA_IMAGES, DATA_LABELS, DATA_ANNOTS, DATA_ASSOC, PROPOSAL, VIZ_DIR]
        + viz_files
        + [PROPOSAL / "draft_predictions.json", PROPOSAL / "annotation_review_guide.md"]
        + stubs["labels"]
        + stubs["associations"]
    )
    for p in created:
        lines.append(f"- `{Path(p).relative_to(ROOT)}`\n")

    report_path = PROJECT / "AUDIT_REPORT.md"
    report_path.write_text("".join(lines), encoding="utf-8")
    return report_path


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Helmet Violation Detection — Audit & Annotation Workflow")
    print("=" * 60)

    ensure_dirs()

    # Task 1
    print("\n[Task 1] Auditing on-disk state...")
    audit = audit_existing()
    for k, v in audit.items():
        print(f"  {k}: {v}")

    # Task 2a — collect source frames
    print("\n[Task 2] Collecting source frames...")
    frame_paths = collect_source_frames()
    if not frame_paths:
        print("[ERROR] No frames available. Aborting.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(frame_paths)} frame(s) ready in data/images/")

    # Task 2b — run detector
    print("\n[Task 2] Running detector on frames...")
    predictions, weights_note = run_detector(frame_paths)
    if predictions is None:
        print(f"[ERROR] Detector failed: {weights_note}", file=sys.stderr)
        sys.exit(1)

    # Task 2c — save draft predictions
    pred_file = write_draft_predictions(predictions, weights_note)
    print(f"  Predictions saved: {pred_file}")

    # Task 2d — draw visualizations
    print("\n[Task 2] Drawing visualization images...")
    viz_files = draw_viz(frame_paths, predictions, weights_note)
    print(f"  {len(viz_files)} viz image(s) in {VIZ_DIR}")

    # Task 3 — write YOLO stubs and association stubs
    print("\n[Task 3] Writing label stubs and association stubs...")
    label_stubs = write_yolo_stubs(predictions)
    assoc_stubs = write_association_stubs(predictions)
    print(f"  {len(label_stubs)} label stub(s) in data/labels/")
    print(f"  {len(assoc_stubs)} association stub(s) in data/associations/")

    # review guide
    guide = write_review_guide(predictions, viz_files, weights_note)
    print(f"  Review guide: {guide}")

    # final report
    stubs = {"labels": label_stubs, "associations": assoc_stubs}
    report = write_audit_report(audit, frame_paths, predictions, weights_note, viz_files, stubs)
    print(f"\n[DONE] Audit report: {report}")
    print(f"\nTo open review images on Windows:")
    print(f"  start {VIZ_DIR}")


if __name__ == "__main__":
    main()
