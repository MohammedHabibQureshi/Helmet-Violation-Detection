# Storage Audit — 2026-10-15
Root: D:\hp\Documents\Claude-workspace\Helmet-Violation-Detection
Actual size: ~608 MB (video already removed; ~593M without video verified)

## Folder sizes (top-level)
- helmet_anpr_research_rebuilt/: 260M (project; includes weights, outputs, backups inside)
- backups/: 217M (pre_phase1_20261015_100529 — duplicate images + old logs)
- runs/: 47M (training artifacts; weights best.pt/last.pt inside)
- weights/: 5.3M (yolo26n.pt + other small weights)
- __pycache__/: 57K (already quarantined in cleanup)
- Other files (zips, .pt at root): ~79M

## 30 largest files (actual paths, sizes)
- 54M helmet_final_audit.zip (DUPLICATE / SUPERSEDED audit archive — safe to remove)
- 22M yolov8s.pt (root) — pretrained fallback; keep
- 22M runs/detect/.../prototype_yolov8s_cpu/weights/last.pt — training artifact; keep
- 22M runs/detect/.../prototype_yolov8s_cpu/weights/best.pt — training artifact; keep
- 22M helmet_anpr_research_rebuilt/yolov8s.pt — rebuilt copy; keep
- 22M helmet_anpr_research_rebuilt/weights/best.pt — CUSTOM CHECKPOINT (must preserve)
- 5.3M weights/yolo26n.pt — pretrained; keep
- 4.3M helmet_anpr_research_code.zip (DUPLICATE / source archive — safe to remove)
- 3.0M outputs/review/label_overlay_contact_sheet.jpg (visual; keep)
- 1.6M-1.5M dataset images (frame_*.jpg) — PRIMARY DATA (must keep)
- 1.5M segment_inference frames — results (keep)
- Backups duplicate images at ~1.5M each (217M total — supersede by active dataset)

## Category classification
MUST PRESERVE (research evidence / code):
- weights/best.pt, src/, data/, results/, reports/, paper/, deliverables/
- dataset.yaml, dataset images/labels, associations.csv, benchmark files
- manuscript (main.tex), bibliography, regenerated tables
- environment diagnostics, cleanup reports

SAFE TO REMOVE (duplicate/superseeded):
- helmet_final_audit.zip (54M) — superseded by deliverables + reports
- helmet_anpr_research_code.zip (4.3M) — source already in rebuilt/
- backups/ (217M) — only if active dataset confirmed intact (yes; verified)
- runs/ training curves/logs (47M) — reproducible from config; keep weights only

UNCERTAIN / KEEP:
- outputs/review/ (viz contacts, ~30M) — not referenced by manuscript directly but preserved
- yolov8s.pt copies — fallback; not deleted

## Verification of required artifacts
- best.pt: PRESENT (22M)
- dataset.yaml: PRESENT
- main.tex: PRESENT
- results/benchmark_runs.csv: PRESENT
- results/associations.csv: PRESENT
- deliverables/FINAL_RESULTS_REPORT.md: PRESENT
- reports/CLEANUP_REPORT.md: PRESENT

## Recommendation (not executed — per instruction: do not delete/yet)
1. Remove helmet_final_audit.zip + helmet_anpr_research_code.zip (~58M reclaim, safe)
2. Remove backups/ after confirming active dataset intact (~217M reclaim, safe if backup not needed)
3. Keep all weights, source, manuscript, results, dataset
4. Do NOT delete video (already removed), dataset images, or model weights
