"""Annotation validator for the Helmet Violation Detection project.

Validates YOLO-format annotation files against expected constraints and
reports errors with line-level detail.
"""

import csv
import json
import sys
from pathlib import Path

from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Valid class IDs
VALID_CLASS_IDS = {0, 1, 2, 3}
CLASS_NAMES = {0: "rider", 1: "with_helmet", 2: "without_helmet", 3: "number_plate"}

# Validation thresholds
MIN_BOX_SIZE = 0.0005  # Minimum width or height in normalized coords
MAX_BOX_SIZE = 1.0     # Maximum width or height
PLATE_MIN_ASPECT = 0.1
PLATE_MAX_ASPECT = 10.0


def validate_annotation_file(label_path: Path) -> list[dict]:
    """Validate a single YOLO annotation file.

    Parameters
    ----------
    label_path:
        Path to the .txt label file.

    Returns
    -------
    list of dict
        Each dict is an error/warning: {line, level, message, raw_line}
    """
    errors = []
    if not label_path.exists():
        return [{"line": None, "level": "error",
                 "message": f"File not found: {label_path}", "raw_line": None}]

    lines = label_path.read_text(encoding="utf-8").splitlines()
    seen_boxes = []  # For duplicate detection

    for line_num, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue  # Skip blank lines and comments silently

        parts = stripped.split()
        if len(parts) != 5:
            errors.append({
                "line": line_num, "level": "error",
                "message": f"Expected 5 fields, got {len(parts)}",
                "raw_line": raw_line,
            })
            continue

        try:
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError as exc:
            errors.append({
                "line": line_num, "level": "error",
                "message": f"Non-numeric value: {exc}",
                "raw_line": raw_line,
            })
            continue

        if class_id not in VALID_CLASS_IDS:
            errors.append({
                "line": line_num, "level": "error",
                "message": f"Invalid class_id={class_id}; "
                           f"valid={sorted(VALID_CLASS_IDS)}",
                "raw_line": raw_line,
            })

        for name, val in [("x", x), ("y", y), ("w", w), ("h", h)]:
            if not (0.0 <= val <= MAX_BOX_SIZE):
                errors.append({
                    "line": line_num, "level": "error",
                    "message": f"Coordinate {name}={val:.6f} out of [0,1]",
                    "raw_line": raw_line,
                })

        if w < MIN_BOX_SIZE:
            errors.append({
                "line": line_num, "level": "error",
                "message": f"Width={w:.6f} below minimum ({MIN_BOX_SIZE})",
                "raw_line": raw_line,
            })
        if h < MIN_BOX_SIZE:
            errors.append({
                "line": line_num, "level": "error",
                "message": f"Height={h:.6f} below minimum ({MIN_BOX_SIZE})",
                "raw_line": raw_line,
            })

        # Plate aspect ratio check
        if class_id == 3 and h > 0:
            aspect = w / h
            if not (PLATE_MIN_ASPECT <= aspect <= PLATE_MAX_ASPECT):
                errors.append({
                    "line": line_num, "level": "warning",
                    "message": f"Plate aspect ratio {aspect:.2f} outside "
                               f"[{PLATE_MIN_ASPECT}, {PLATE_MAX_ASPECT}]",
                    "raw_line": raw_line,
                })

        # Detect exact duplicates
        box_key = (class_id, round(x, 5), round(y, 5), round(w, 5), round(h, 5))
        if box_key in seen_boxes:
            errors.append({
                "line": line_num, "level": "warning",
                "message": f"Duplicate annotation box at line {line_num}",
                "raw_line": raw_line,
            })
        else:
            seen_boxes.append(box_key)

    return errors


def validate_dataset(
    labels_dir: Path,
    images_dir: Path = None,
    report_path: Path = None,
) -> dict:
    """Validate all annotation files in a directory.

    Parameters
    ----------
    labels_dir:
        Directory containing YOLO .txt label files.
    images_dir:
        Optional image directory; checks for orphan labels / missing labels.
    report_path:
        Path to save the JSON validation report.

    Returns
    -------
    dict
        Validation summary including per-file errors.
    """
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        return {
            "status": "error",
            "message": f"Labels directory not found: {labels_dir}",
            "files_checked": 0,
            "errors_total": 0,
            "warnings_total": 0,
            "file_results": [],
        }

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        return {
            "status": "warning",
            "message": "No .txt label files found",
            "files_checked": 0,
            "errors_total": 0,
            "warnings_total": 0,
            "file_results": [],
        }

    file_results = []
    total_errors = 0
    total_warnings = 0
    total_annotations = 0
    class_counts = {cid: 0 for cid in VALID_CLASS_IDS}

    for label_path in label_files:
        issues = validate_annotation_file(label_path)
        errors = [i for i in issues if i["level"] == "error"]
        warnings = [i for i in issues if i["level"] == "warning"]
        total_errors += len(errors)
        total_warnings += len(warnings)

        # Count annotations by class
        ann_count = 0
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        cid = int(parts[0])
                        if cid in class_counts:
                            class_counts[cid] += 1
                        ann_count += 1
                    except ValueError:
                        pass
        total_annotations += ann_count
        file_results.append({
            "file": str(label_path),
            "annotation_count": ann_count,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "issues": issues,
        })

    # Orphan label check
    orphan_labels = []
    missing_labels = []
    if images_dir is not None:
        images_dir = Path(images_dir)
        if images_dir.exists():
            image_stems = {p.stem for p in images_dir.glob("*")
                           if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")}
            label_stems = {p.stem for p in label_files}
            orphan_labels = sorted(label_stems - image_stems)
            missing_labels = sorted(image_stems - label_stems)

    report = {
        "status": "passed" if total_errors == 0 else "failed",
        "files_checked": len(label_files),
        "total_annotations": total_annotations,
        "errors_total": total_errors,
        "warnings_total": total_warnings,
        "class_distribution": {
            CLASS_NAMES.get(cid, str(cid)): count
            for cid, count in class_counts.items()
        },
        "orphan_labels": orphan_labels,
        "missing_labels": missing_labels,
        "file_results": file_results,
    }

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    return report


def main() -> int:
    """Entry point."""
    labels_dir = DATA_DIR / "labels"
    images_dir = DATA_DIR / "images"
    report_path = RESULTS_DIR / "annotation_validation.json"
    print(f"Validating annotations in: {labels_dir}")

    report = validate_dataset(labels_dir, images_dir=images_dir, report_path=report_path)
    print(f"Files checked:      {report['files_checked']}")
    print(f"Total annotations:  {report['total_annotations']}")
    print(f"Errors:             {report['errors_total']}")
    print(f"Warnings:           {report['warnings_total']}")
    print(f"Status:             {report['status']}")
    if report_path:
        print(f"Report saved to: {report_path}")
    return 0 if report["status"] in ("passed", "warning") else 1


if __name__ == "__main__":
    sys.exit(main())
