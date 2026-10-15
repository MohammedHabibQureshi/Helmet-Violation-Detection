# REMAINING MANUAL REVIEW — What Only User Can Complete
Date: 2026-10-14
Status: BLOCKED — not skipped; explicitly documented

## Required user actions (not automatable)
1. Open annotate.py: `python -I annotate.py --start 0`
2. For each of 40 frames: verify COCO boxes; draw rider boxes (LMB drag); assign class (1-4 keys); confirm helmet (A/L) with association; enter plate text (T); save (SPACE); verify (V) until 0 issues.
3. After 40 frames: confirm associations file per frame; confirm label files have verified boxes only.

## Blocked outputs (awaiting above)
- Verified associations (currently PENDING in 40 files).
- Verified plate text (currently NONE confirmed).
- Research-grade dataset gate (cannot declare passed without verified associations + labels).
- Research-grade evaluation (current metrics are prototype-only with unverified GT).

## What IS complete without user
- All tool fixes, tests, backups, split, prototype training, benchmark CSV, reports.

## Do not claim pipeline complete until above done.
