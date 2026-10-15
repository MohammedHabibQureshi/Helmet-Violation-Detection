"""Repair the documented GUI persistence defect that shifted every class ID by +1.

The script is deliberately guarded: it runs only when the complete label corpus uses
IDs 1..4 and no ID 0, then writes a manifest with before/after hashes.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "labels"
OUT = ROOT / "reports" / "LABEL_CLASS_SHIFT_REPAIR.json"

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

paths = sorted(LABELS.glob("*.txt"))
ids = {int(line.split()[0]) for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
if not ids.issubset({1, 2, 3, 4}) or 0 in ids:
    raise SystemExit(f"Refusing repair: expected only shifted IDs 1..4, found {sorted(ids)}")

changes = []
for path in paths:
    before = digest(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        fields[0] = str(int(fields[0]) - 1)
        rows.append(" ".join(fields))
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    changes.append({"file": str(path.relative_to(ROOT)).replace("\\", "/"), "before_sha256": before,
                    "after_sha256": digest(path), "boxes": len(rows)})

OUT.write_text(json.dumps({
    "status": "REPAIRED",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "defect": "GUI persistence incremented each project class ID by one",
    "mapping": {"1": 0, "2": 1, "3": 2, "4": 3},
    "files": changes,
}, indent=2), encoding="utf-8")
print(f"repaired {len(changes)} files; manifest: {OUT}")
