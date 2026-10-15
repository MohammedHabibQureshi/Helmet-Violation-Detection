# Helmet Violation Detection — Annotation Guidelines

## 1. Purpose and Scope

This document defines the annotation standards for the Helmet Violation Detection and Automatic Number Plate Recognition (ANPR) dataset. The dataset is annotated with four object classes to support two concurrent tasks: (a) detection of motorcyclist riders and their helmet status (wearing a helmet vs. not wearing a helmet), and (b) detection and recognition of vehicle number plates. These annotations are used to train, validate, and evaluate an end-to-end system that links a detected helmetless rider to a nearby vehicle number plate for violation attribution.

All annotations are stored in YOLO format (one text file per image, with lines `class_id x_center y_center width height`, all normalized to the [0, 1] image coordinate space). Every annotator must read this document in full before starting and must adhere to the rules consistently. Inconsistency between annotators is a greater source of error than small subjective deviations; therefore, all borderline cases are resolved by consensus following the guidance below.

## 2. Class Definitions

### 2.1 `rider` (class id 0)
A human rider of a two-wheeled vehicle (motorcycle, moped, scooter). Annotate the rider as a standing/riding person, not the vehicle itself. The bounding box should tightly enclose the rider's visible body excluding the vehicle.

### 2.2 `with_helmet` (class id 1)
A rider whose head is covered by a helmet. The helmet must be fully visible and identifiable as a helmet (rounded shell, eye opening, strap visible where present). This class is mutually exclusive with `without_helmet` on the same rider at the same timestamp: a rider is either annotated with a helmet class or the rider class (see below).

### 2.3 `without_helmet` (class id 2)
A rider whose head is bare (no helmet, no cap, no cloth head covering that conceals the hairline) and who is clearly riding or about to ride a two-wheeled vehicle in traffic. Head covered by a non-helmet item (baseball cap, turban, scarf) is annotated as `without_helmet` unless it unmistakably resembles a standard safety helmet.

### 2.4 `number_plate` (class id 3)
The rectangular registration plate of the vehicle the rider is operating. Annotate the full visible plate area. If only part of the plate is visible, annotate the visible rectangular region and mark the annotation as partially occluded (see Section 5).

## 3. Bounding Box Conventions

1. **YOLO format**: Every line is `class_id x y w h` where x, y, w, h are normalized coordinates in [0, 1]. x and y are the center coordinates; w and h are the width and height.
2. **Minimal tight boxes**: Boxes must be tight to the visible extent of the annotated entity. Extra whitespace inside a box is considered an error. Do not expand boxes to include adjacent objects.
3. **Tolerance for small bodies**: For very distant riders whose body occupies fewer than ~30 pixels in height, allow a single-pixel tolerance but do not pad boxes beyond the visible body.
4. **Plate rotation**: Number plates on motorcycles are typically rotated in 3D. Annotate the axis-aligned (corner-aligned) bounding box that tightly encloses the visible plate rectangle.
5. **No overlap between rider and helmet classes**: Do not annotate both `with_helmet` and `without_helmet` on the same rider. Do not annotate a separate `rider` box when a helmet class box tightly covers the same person; use the helmet class box only.

## 4. Spatial Relationships and Association Semantics

The downstream system associates a `without_helmet` box with a nearby `number_plate` box. Annotators should keep this in mind:

- The plate associated with a rider is the plate of the vehicle that rider is physically occupying.
- Do not merge a plate onto the rider's body; plates must be annotated separately on the vehicle's rear/front trunk area.
- When a single plate is visible and clearly belongs to the rider's vehicle, annotate it even if the rider is partially occluded.

## 5. Edge Cases

### 5.1 Occluded riders
If a rider is occluded by another object (vehicle, pedestrian, sign), annotate the rider only if more than 50% of the rider's body surface is visible and the head is visible. If the head is occluded but clearly not wearing a helmet, label `without_helmet`; if the head is obscured and helmet status cannot be determined, do not annotate that rider at all — do not guess.

### 5.2 Multiple riders on one vehicle
If two riders are seated on the same motorcycle (pillion passenger), annotate each rider separately. Annotate `with_helmet`/`without_helmet` per rider. This is the most common real-world violation scenario.

### 5.3 Partial number plates
If only a portion of the plate is visible due to occlusion, wheel, or bodywork, annotate the visible rectangular region and set the annotation attribute `occluded=true` (stored in a companion attribute file). Do not extrapolate the full plate geometry.

### 5.4 No helmet, rider absent
If a vehicle appears without any visible rider (parked, rider not in frame), do not annotate a rider. Only annotate the plate if it is clearly legible and belongs to the visible vehicle.

### 5.5 Side-view riders
For side views where the rider's back faces the camera, the head may appear smaller. Helmet status should still be assigned from any visible part of the head. When uncertain from a single view and a multi-frame sequence is available, resolve using adjacent frames.

### 5.6 Groups of vehicles
Annotate every visible rider and plate that meets the visibility criteria. Do not skip small or distant objects if they meet the threshold; however, objects smaller than ~15 pixels in width on any side are considered out-of-scope and may be skipped with the `too_small` attribute.

## 6. Quality Standards

1. **Consistency**: Apply the same rules to every frame, regardless of lighting or motion blur.
2. **Tightness**: Boxes must not extend beyond the visible extent of the annotated entity.
3. **Correct class assignment**: Misclassifying a helmeted rider as `without_helmet`, or vice versa, is a critical error.
4. **No duplicate boxes**: Do not create nested boxes for the same object (e.g., one box per shoulder).
5. **Confidence**: Mark any annotation whose correctness cannot be established from the visual evidence with the `uncertain` attribute.

## 7. Handling Ambiguous Cases

When a case is genuinely ambiguous:

1. Consult the adjacent frames (video context) — the majority view across 3–5 frames is preferred.
2. Prefer the conservative choice: when in doubt whether a head covering is a helmet, classify as `without_helmet` (conservative for a safety violation detector).
3. When rider vs. helmet-class choice is unclear, default to annotating `rider` without a helmet class.
4. Log the ambiguity in the annotation notes and resolve by team consensus at the end of each batch.

## 8. Export Format

### 8.1 Primary format — YOLO
Each image has a corresponding `.txt` label file with the same basename. Example:

```
2 0.4231 0.6104 0.0512 0.1305
0 0.8912 0.4456 0.0923 0.2134
3 0.4389 0.7012 0.0601 0.0289
```

### 8.2 Optional attributes file (JSON)
Per-image JSON with attributes:

```json
{
  "image": "frame_001234.jpg",
  "annotations": [
    {"class_id": 2, "occ": true, "reason": "head_covered_by_scarf"},
    {"class_id": 3, "occ": false}
  ],
  "ambiguous": ["frame_001234"]
}
```

### 8.3 Sequence/association metadata
Per-video TSV mapping plate detections to rider detections with an association confidence (computed downstream, annotated by a human validator when ground-truth association is known).

## 9. Validation Checklist

Before submitting an annotation batch, the annotator must verify:

- [ ] All YOLO lines have exactly 5 numeric fields (class_id, x, y, w, h)
- [ ] All coordinates are within [0.0, 1.0]
- [ ] w and h are greater than 0.0005 (no degenerate boxes)
- [ ] No frame that should have annotations is missing its label file
- [ ] Rider/helmet classes are mutually exclusive per rider
- [ ] No nested/duplicate boxes for the same object
- [ ] Plate boxes have reasonable aspect ratio (0.3 to 3.0)
- [ ] Any partially occluded object is marked with `occ=true`
- [ ] Ambiguous frames are listed in the ambiguous list
- [ ] Label file count equals image file count for the batch

Run the automated validation script (`src/annotation_validator.py`) before submission; fix all reported errors.
