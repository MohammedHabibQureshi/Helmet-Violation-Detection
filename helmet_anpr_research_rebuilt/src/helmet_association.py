"""Helmet-to-plate association module.

Links detections of helmetless riders to nearby number plates using
spatial proximity and IoU-based containment rules.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AssociationConfig:
    """Thresholds for the helmet-plate association logic."""

    rider_helmet_iou_threshold: float = 0.05
    """Min IoU between a rider box and a without_helmet box to be considered
    the same person's head region overlap."""

    rider_containment_threshold: float = 0.4
    """Min fraction of the without_helmet box that must lie inside the
    rider box (containment instead of IoU)."""

    plate_vertical_search_frac: float = 0.6
    """How far below the rider box bottom (as fraction of rider box height)
    to search for a plate."""

    plate_horizontal_overlap_frac: float = 0.3
    """Min horizontal overlap fraction between the rider box and the plate
    box for the plate to be considered associated."""

    min_plate_area: float = 100.0
    """Minimum plate bounding box area in pixels^2."""

    max_association_distance_px: float = 400.0
    """Maximum euclidean distance between rider centre and plate centre."""


def _bbox_iou(box_a: list, box_b: list) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area == 0.0:
        return 0.0
    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _containment(box_inner: list, box_outer: list) -> float:
    """Fraction of box_inner that lies inside box_outer."""
    x1 = max(box_inner[0], box_outer[0])
    y1 = max(box_inner[1], box_outer[1])
    x2 = min(box_inner[2], box_outer[2])
    y2 = min(box_inner[3], box_outer[3])
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    inner_area = max(1e-9, (box_inner[2] - box_inner[0]) * (box_inner[3] - box_inner[1]))
    return inter_area / inner_area


def _box_center(box: list) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _euclidean(c1: tuple, c2: tuple) -> float:
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def _horizontal_overlap_fraction(box_a: list, box_b: list) -> float:
    """Fraction of box_a's horizontal span that overlaps with box_b."""
    overlap = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    width_a = max(1e-9, box_a[2] - box_a[0])
    return overlap / width_a


def associate_violations(
    detections: list[dict],
    frame_idx: int = 0,
    timestamp_s: float = 0.0,
    config: AssociationConfig = None,
) -> list[dict]:
    """Associate helmetless rider detections with number plates.

    Parameters
    ----------
    detections:
        List of detection dicts from ``YOLODetector.detect_frame``.
        Each dict must have: class_name, conf, x1, y1, x2, y2.
    frame_idx:
        Frame index for logging.
    timestamp_s:
        Timestamp for the frame, in seconds.
    config:
        Association thresholds. Uses defaults if None.

    Returns
    -------
    list of dict
        ViolationCandidate dicts, one per (rider, without_helmet, plate)
        triplet evaluated.  Both accepted and rejected candidates are
        returned; check the ``accepted`` field.
    """
    cfg = config or AssociationConfig()

    riders = [
        d for d in detections if d.get("class_name") == "rider"
    ]
    no_helmets = [
        d for d in detections if d.get("class_name") == "without_helmet"
    ]
    plates = [
        d for d in detections if d.get("class_name") == "number_plate"
    ]

    candidates = []

    for nh in no_helmets:
        nh_box = [nh["x1"], nh["y1"], nh["x2"], nh["y2"]]

        # Find the rider box that best contains this without_helmet box
        best_rider = None
        best_rider_score = -1.0
        for rider in riders:
            r_box = [rider["x1"], rider["y1"], rider["x2"], rider["y2"]]
            containment = _containment(nh_box, r_box)
            iou = _bbox_iou(nh_box, r_box)
            score = max(containment, iou)
            if (
                containment >= cfg.rider_containment_threshold
                or iou >= cfg.rider_helmet_iou_threshold
            ) and score > best_rider_score:
                best_rider_score = score
                best_rider = rider

        # If no rider found, use the without_helmet box as the rider proxy
        if best_rider is None:
            rider_box_used = nh_box
            rider_association_note = "no_rider_box_found"
        else:
            rider_box_used = [
                best_rider["x1"], best_rider["y1"],
                best_rider["x2"], best_rider["y2"]
            ]
            rider_association_note = "rider_box_matched"

        # Find the nearest plate below/near the rider box
        rider_cy = (rider_box_used[1] + rider_box_used[3]) / 2.0
        rider_height = max(1.0, rider_box_used[3] - rider_box_used[1])
        rider_cx = (rider_box_used[0] + rider_box_used[2]) / 2.0

        # Search region: extend downward from rider bottom
        search_bottom = rider_box_used[3] + cfg.plate_vertical_search_frac * rider_height

        best_plate = None
        best_plate_score = float("inf")
        rejection_reason = "no_plate_in_search_region"

        for plate in plates:
            p_box = [plate["x1"], plate["y1"], plate["x2"], plate["y2"]]
            p_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])

            if p_area < cfg.min_plate_area:
                continue

            p_cx, p_cy = _box_center(p_box)
            dist = _euclidean((rider_cx, rider_cy), (p_cx, p_cy))

            # Plate must be within vertical search band
            if p_cy > search_bottom:
                continue
            if dist > cfg.max_association_distance_px:
                continue

            h_overlap = _horizontal_overlap_fraction(rider_box_used, p_box)
            if h_overlap < cfg.plate_horizontal_overlap_frac:
                continue

            # Score: prefer closer plates
            if dist < best_plate_score:
                best_plate_score = dist
                best_plate = plate

        # Build candidate
        accepted = best_plate is not None
        if not accepted and rejection_reason == "no_plate_in_search_region":
            # Provide more specific rejection reasons
            if not plates:
                rejection_reason = "no_plates_detected_in_frame"
            else:
                # Check if any plate was too small or too far
                for plate in plates:
                    p_box = [plate["x1"], plate["y1"], plate["x2"], plate["y2"]]
                    p_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                    dist = _euclidean(
                        (rider_cx, rider_cy), _box_center(p_box)
                    )
                    if p_area < cfg.min_plate_area:
                        rejection_reason = "plate_too_small"
                    elif dist > cfg.max_association_distance_px:
                        rejection_reason = "plate_too_far"
                    else:
                        rejection_reason = "plate_outside_search_window"

        candidate = {
            "frame_idx": frame_idx,
            "timestamp_s": timestamp_s,
            "rider_box": (
                [best_rider["x1"], best_rider["y1"],
                 best_rider["x2"], best_rider["y2"]]
                if best_rider else None
            ),
            "helmet_box": nh_box,
            "plate_box": (
                [best_plate["x1"], best_plate["y1"],
                 best_plate["x2"], best_plate["y2"]]
                if best_plate else None
            ),
            "plate_confidence": best_plate["conf"] if best_plate else None,
            "rider_association_score": round(best_rider_score, 4),
            "rider_association_note": rider_association_note,
            "plate_distance_px": (
                round(best_plate_score, 2) if best_plate else None
            ),
            "association_score": (
                round(
                    (best_rider_score + 1.0 / (1.0 + best_plate_score)) / 2.0, 4
                )
                if best_plate else 0.0
            ),
            "accepted": accepted,
            "rejection_reason": None if accepted else rejection_reason,
        }
        candidates.append(candidate)

    return candidates


def main() -> int:
    """Smoke-test with synthetic detections."""
    detections = [
        {"class_name": "rider", "conf": 0.9,
         "x1": 100, "y1": 50, "x2": 200, "y2": 300},
        {"class_name": "without_helmet", "conf": 0.85,
         "x1": 120, "y1": 55, "x2": 180, "y2": 100},
        {"class_name": "number_plate", "conf": 0.8,
         "x1": 110, "y1": 310, "x2": 190, "y2": 340},
    ]
    results = associate_violations(detections, frame_idx=0, timestamp_s=0.0)
    for c in results:
        status = "ACCEPTED" if c["accepted"] else f"REJECTED ({c['rejection_reason']})"
        print(f"Candidate: {status} | plate_box={c['plate_box']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
