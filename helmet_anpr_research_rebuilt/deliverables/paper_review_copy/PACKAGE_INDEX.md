# paper_review.zip — Audit Package Index
Created: 2026-10-15 | Project: helmet_anpr_research_rebuilt

## INCLUDED (verified present in ZIP)
- paper/main.tex (main manuscript)
- paper/main_legacy_uncompiled.tex (legacy version)
- paper/references.bib
- paper/tables/table_iii_regenerated.tex
- paper/tables/table_viii_regenerated.tex
- paper/figures/figure_manifest.md (existing, preserved)
- results/benchmark_runs.csv
- results/benchmark_summary.json
- results/raw_detection_predictions.json
- results/violation_events.json
- results/associations.csv
- deliverables/FINAL_RESULTS_REPORT.md
- deliverables/COMPLETION_TASK_LEDGER.json
- deliverables/ARTIFACT_MANIFEST.json
- reports/ENVIRONMENT_DIAGNOSTICS.md
- reports/CUDA_DIAGNOSTICS.md
- reports/DATASET_VALIDATION.json
- reports/ANNOTATION_ONTOLOGY.md
- reports/DATASET_REMEDIATION_REPORT.md
- data/dataset.yaml
- data/splits/*.txt (split manifests)
- configs/*.yaml (config files)
- src/association.py (stable-ID association module)

## REQUESTED BUT MISSING (documented, not fabricated)
- run_metadata.json — NOT PRODUCED (pipeline run metadata never generated)
- figures/confusion_matrix.png — NOT PRODUCED (requires validated GT)
- figures/precision_recall_curves.png — NOT PRODUCED (requires validated GT)
- figures/training_curves.png — NOT PRODUCED (training not completed against verified labels)
- outputs/annotated_video.mp4 — EXCLUDED per instructions (large video)
- paper/figures/*.png — EXCLUDED (original dataset images excluded per instructions)
- weights/best.pt — EXCLUDED (large model weight; metadata only not present)
- README / reproduction instructions — NOT PRESENT in original; not fabricated

## EXCLUDED PER INSTRUCTIONS
- Original video (croped-main.mp4)
- Raw image dataset (40 .jpg frames)
- Virtual environments / caches
- Git history
- Large generated videos
- Model weights (except if metadata needed; none produced)
