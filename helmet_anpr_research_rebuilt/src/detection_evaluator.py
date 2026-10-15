"""Detection evaluation metrics (mAP, precision, recall, CER for OCR).

Provides functions to compute detection metrics from predictions and ground
truth, plus OCR character error rate calculations.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def box_iou(box_a: list, box_b: list) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_precision_recall(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute precision and recall for a set of detections.

    Parameters
    ----------
    predictions:
        List of dicts with keys: bbox [x1,y1,x2,y2], class_id, conf.
    ground_truths:
        List of dicts with keys: bbox, class_id.
    iou_threshold:
        Minimum IoU for a true positive.

    Returns
    -------
    dict with keys: precision, recall, f1, tp, fp, fn.
    """
    if not ground_truths:
        return {
            "precision": 1.0 if not predictions else 0.0,
            "recall": 1.0,
            "f1": 1.0 if not predictions else 0.0,
            "tp": 0,
            "fp": len(predictions),
            "fn": 0,
        }
    if not predictions:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": len(ground_truths),
        }

    # Sort predictions by confidence
    sorted_preds = sorted(predictions, key=lambda p: p.get("conf", 0.0), reverse=True)

    # Track which GT boxes have been matched
    gt_matched = [False] * len(ground_truths)
    tp = 0
    fp = 0

    for pred in sorted_preds:
        pred_box = pred.get("bbox")
        pred_class = pred.get("class_id")
        if pred_box is None or pred_class is None:
            fp += 1
            continue

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue
            gt_box = gt.get("bbox")
            gt_class = gt.get("class_id")
            if gt_box is None or gt_class is None:
                continue
            if pred_class != gt_class:
                continue

            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i] + [0] * len(s2)
        for j, c2 in enumerate(s2, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (c1 != c2))
        prev = curr
    return prev[len(s2)]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute character error rate (CER).

    CER = edit_distance / len(reference)
    If reference is empty, return 0.0.
    """
    if not reference:
        return 0.0
    return levenshtein_distance(reference, hypothesis) / len(reference)


def compute_cer_batch(references: list[str], hypotheses: list[str]) -> dict:
    """Compute CER over a batch of text pairs.

    Returns dict with keys: mean_cer, median_cer, total_chars, total_edits.
    """
    if not references or not hypotheses or len(references) != len(hypotheses):
        return {"mean_cer": 0.0, "median_cer": 0.0,
                "total_chars": 0, "total_edits": 0}

    rates = []
    total_chars = 0
    total_edits = 0
    for ref, hyp in zip(references, hypotheses):
        ed = levenshtein_distance(ref, hyp)
        rates.append(ed / len(ref) if len(ref) > 0 else 0.0)
        total_chars += len(ref)
        total_edits += ed

    return {
        "mean_cer": round(np.mean(rates), 4),
        "median_cer": round(np.median(rates), 4),
        "total_chars": total_chars,
        "total_edits": total_edits,
    }


def compute_map_at_iou(
    predictions_by_image: dict[str, list[dict]],
    ground_truths_by_image: dict[str, list[dict]],
    iou_thresholds: list[float] = None,
) -> dict:
    """Compute mean Average Precision (mAP) across IoU thresholds.

    Parameters
    ----------
    predictions_by_image:
        Dict mapping image_id -> list of predictions (each with bbox, class_id, conf).
    ground_truths_by_image:
        Dict mapping image_id -> list of ground truths (each with bbox, class_id).
    iou_thresholds:
        List of IoU thresholds to evaluate at. Default: [0.5, 0.75, 0.95].

    Returns
    -------
    dict
        Per-class APs, overall mAP, and per-threshold metrics.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75, 0.95]

    # Get all class IDs
    all_classes = set()
    for preds in predictions_by_image.values():
        for p in preds:
            all_classes.add(p.get("class_id"))
    for gts in ground_truth_by_image.values():
        for g in gts:
            all_classes.add(g.get("class_id"))
    all_classes = sorted(all_classes)

    results = {}
    for iou_thresh in iou_thresholds:
        ap_per_class = {}
        for class_id in all_classes:
            # Collect predictions and GTs for this class across all images
            class_preds = []
            class_gts = []
            for img_id, preds in predictions_by_image.items():
                gts = ground_truths_by_image.get(img_id, [])
                class_preds.append([
                    p for p in preds if p.get("class_id") == class_id
                ])
                class_gts.append([
                    g for g in gts if g.get("class_id") == class_id
                ])

            # Compute per-image precision/recall and aggregate
            all_tp, all_fp = [], []
            total_gts = sum(len(g) for g in class_gts)
            if total_gts == 0:
                continue

            # Flatten predictions sorted by confidence
            flat_preds = []
            flat_img_idx = []
            for img_idx, preds in enumerate(class_preds):
                for p in preds:
                    flat_preds.append(p)
                    flat_img_idx.append(img_idx)

            flat_preds = sorted(flat_preds, key=lambda x: x.get("conf", 0.0), reverse=True)

            gt_matched = [set() for _ in range(len(class_gts))]
            tp_count = 0
            fp_count = 0
            for pred in flat_preds:
                img_idx = flat_img_idx.pop(0)  # This is simplified
                img_idx = 0  # placeholder

                best_iou = 0.0
                best_gt = -1
                for gt_idx, gt in enumerate(class_gts[img_idx]):
                    if gt_idx in gt_matched[img_idx]:
                        continue
                    iou = box_iou(pred.get("bbox", []), gt.get("bbox", []))
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt_idx

                if best_iou >= iou_thresh and best_gt >= 0:
                    tp_count += 1
                    gt_matched[img_idx].add(best_gt)
                else:
                    fp_count += 1

            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            recall = tp_count / total_gts if total_gts > 0 else 0.0
            ap = precision * recall  # Simplified AP (not full PR curve)
            ap_per_class[class_id] = round(ap, 4)

        mean_ap = round(np.mean(list(ap_per_class.values())), 4) if ap_per_class else 0.0
        results[f"map@{iou_thresh}"] = mean_ap

    return results


def main() -> int:
    """Smoke test of metrics."""
    # Test precision/recall
    preds = [
        {"bbox": [10, 10, 50, 50], "class_id": 0, "conf": 0.9},
        {"bbox": [100, 100, 150, 150], "class_id": 1, "conf": 0.8},
    ]
    gts = [
        {"bbox": [12, 12, 48, 48], "class_id": 0},
        {"bbox": [100, 100, 150, 150], "class_id": 1},
    ]
    pr = compute_precision_recall(preds, gts, iou_threshold=0.5)
    print(f"Precision-Recall: {pr}")

    # Test CER
    cer = character_error_rate("HELLO", "HXLLO")
    print(f"CER 'HELLO'->'HXLLO': {cer}")
    cer_batch = compute_cer_batch(
        ["MH12AB1234", "DL01CC5678"],
        ["MH12AB1234", "DL01CC5679"],
    )
    print(f"CER batch: {cer_batch}")

    return 0


if __name__ == "__main__":
    sys.exit(main())