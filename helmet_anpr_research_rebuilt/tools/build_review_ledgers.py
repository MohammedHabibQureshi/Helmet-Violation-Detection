"""Create auditable ledgers after the class-shift and crop-source investigation."""
from pathlib import Path
import csv
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "images"
LEGACY = ROOT / "outputs" / "review" / "plate_crops"
LABEL_CROPS = ROOT / "outputs" / "review" / "label_plate_crops"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
stamp = datetime.now(timezone.utc).isoformat()

# The legacy set was visually audited as non-plate crops; it must not be used
# for OCR evaluation. All emitted records preserve the original file path.
with (REPORTS / "LEGACY_PLATE_CROP_AUDIT.csv").open("w", newline="", encoding="utf-8") as f:
    fields = ["legacy_crop", "decision", "reason", "reviewer", "reviewed_at_utc"]
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for p in sorted(LEGACY.glob("*.jpg")):
        w.writerow({"legacy_crop": str(p.relative_to(ROOT)).replace("\\", "/"),
                    "decision": "REJECTED_WRONG_CROP",
                    "reason": "Visual audit: crop does not show a number plate; legacy crop source is not the repaired class-3 YOLO box.",
                    "reviewer": "AI-assisted visual audit", "reviewed_at_utc": stamp})

# One string is unambiguous in the source-derived crop. The other text remains
# unreadable rather than guessed. A human reviewer can replace only an
# UNREADABLE status with evidence-backed transcription.
known = {"frame_00007275_labelidx_0.jpg": "AP39CV2294"}
with (REPORTS / "PLATE_GROUND_TRUTH_AI_ASSISTED.csv").open("w", newline="", encoding="utf-8") as f:
    fields = ["crop", "transcription", "status", "evidence", "reviewer", "reviewed_at_utc"]
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for p in sorted(LABEL_CROPS.glob("*.jpg")):
        text = known.get(p.name, "")
        w.writerow({"crop": str(p.relative_to(ROOT)).replace("\\", "/"),
                    "transcription": text,
                    "status": "AI_VISUALLY_READ" if text else "UNREADABLE",
                    "evidence": "Re-extracted from repaired class-3 box in original source frame; no OCR text promoted.",
                    "reviewer": "AI-assisted visual audit", "reviewed_at_utc": stamp})

empty = {"frame_00006790": "NO_TARGETS_PRESENT", "frame_00012610": "NO_TARGETS_PRESENT",
         "frame_00014065": "NO_TARGETS_PRESENT", "frame_00015035": "NO_TARGETS_PRESENT",
         "frame_00016490": "NO_TARGETS_PRESENT", "frame_00017945": "NO_TARGETS_PRESENT"}
with (REPORTS / "EMPTY_FRAME_AI_ASSISTED_REVIEW.csv").open("w", newline="", encoding="utf-8") as f:
    fields = ["frame", "decision", "basis", "reviewer", "reviewed_at_utc"]
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for frame, decision in empty.items():
        w.writerow({"frame": frame, "decision": decision,
                    "basis": "Native source frame and label-overlay contact-sheet inspection; no clearly assessable active rider/helmet/plate target.",
                    "reviewer": "AI-assisted visual audit", "reviewed_at_utc": stamp})

with (REPORTS / "FRAME_SEMANTIC_REVIEW_QUEUE.csv").open("w", newline="", encoding="utf-8") as f:
    fields = ["frame", "label_boxes", "review_status", "required_action"]
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for p in sorted(IMAGES.glob("*.jpg")):
        label = ROOT / "data" / "labels" / f"{p.stem}.txt"
        n = len([x for x in label.read_text(encoding="utf-8").splitlines() if x.strip()])
        w.writerow({"frame": p.stem, "label_boxes": n, "review_status": "PENDING_INDEPENDENT_SEMANTIC_REVIEW",
                    "required_action": "Confirm box geometry and mutually exclusive helmet class; confirm rider-to-plate ownership."})
print("review ledgers written")
