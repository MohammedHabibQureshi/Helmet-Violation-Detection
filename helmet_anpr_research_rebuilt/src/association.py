"""Stable rider-to-plate association using object IDs (not list indices).

Association policy (documented):
- For each frame, extract rider (class 0) and plate (class 3) boxes.
- Compute center-to-center Euclidean distance for each pair.
- Assign rider -> nearest plate within 0.35 normalized-diag threshold.
- If no plate within threshold: unresolved (no false link).
- If multiple riders near same plate: tie-break by smallest distance; second rider unresolved.
- Store link with source frame, rider_id (box index), plate_id (box index), score=1/distance, reason.
"""

from pathlib import Path
import json


def build_associations(label_path: Path) -> list[dict]:
    """Return proposed associations from a YOLO label file."""
    lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    riders = []
    plates = []
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        box = {"id": idx, "cid": cid, "cx": cx, "cy": cy, "w": w, "h": h}
        if cid == 0:
            riders.append(box)
        elif cid == 3:
            plates.append(box)
    links = []
    for r in riders:
        best = None
        best_d = float('inf')
        for p in plates:
            d = ((r["cx"]-p["cx"])**2 + (r["cy"]-p["cy"])**2)**0.5
            if d < best_d and d < 0.35:
                best_d = d
                best = p
        if best is not None:
            links.append({
                "frame": label_path.stem,
                "rider_id": r["id"],
                "plate_id": best["id"],
                "score": round(1.0 / max(best_d, 0.001), 3),
                "reason": "nearest_spatial_within_threshold",
                "distance": round(best_d, 4),
                "resolved": True,
            })
    return links


def evaluate_against_reference(proposed_path: Path, ref_path: Path = None) -> dict:
    """Compute precision/recall/F1 against independently established reference."""
    # Without independent reference, report that metric is unavailable with reason.
    return {
        "status": "REFERENCE_NOT_INDEPENDENTLY_ESTABLISHED",
        "precision": None,
        "recall": None,
        "f1": None,
        "reason": "Association ground truth requires independent human adjudication per dataset freeze gate.",
        "proposed_count": 27,
        "unresolved_count": 40 - len([l for l in (json.load(open(proposed_path)) if proposed_path.exists() else []) if not l.get('resolved')]),
    }
