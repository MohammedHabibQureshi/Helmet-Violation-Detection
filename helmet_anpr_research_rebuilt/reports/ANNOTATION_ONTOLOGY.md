# Annotation Ontology — Helmet Violation Detection

## Class definitions (4-class YOLO ontology)
- **0 — rider**: Full motorcycle rider (person + motorcycle region, bounding the visible rider body). Not head-only.
- **1 — with_helmet**: Rider whose head region shows a helmet (positive helmet compliance).
- **2 — without_helmet**: Rider whose head region is visible and no helmet is present (violation).
- **3 — number_plate**: License plate region on the motorcycle (visual text region, not necessarily readable at this resolution).

## Design choices
- Rider box (class 0) represents the rider + motorcycle region; it is not a head-only crop.
- Helmet-state boxes (1, 2) are head-level subregions tied to the rider; they are not full-body boxes.
- Plate box (3) is separate from rider; association links rider -> plate via spatial proximity.
- Overlapping definitions are avoided: a rider is class 0; helmet-state is an attribute subclass (1 or 2); plate is a separate object (3).
- Empty-label frames (6) are recorded as negatives; they are not automatically assumed to contain no objects — independent review required.
