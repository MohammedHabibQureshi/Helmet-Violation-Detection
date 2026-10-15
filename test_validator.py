"""
Validator test: runs validate_annotation_file on known-valid and known-invalid inputs.
Expected outcomes are asserted — if any assertion fails the test reports FAIL.

Run from project root:
    python -I test_validator.py
"""

import sys
import tempfile
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "helmet_anpr_research_rebuilt"))
from src.annotation_validator import validate_annotation_file, validate_dataset

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

failures = []

def check(name, issues, expected_errors, expected_warnings):
    errors   = [i for i in issues if i["level"] == "error"]
    warnings = [i for i in issues if i["level"] == "warning"]
    ok = (len(errors) == expected_errors and len(warnings) == expected_warnings)
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}")
    if not ok:
        print(f"         expected errors={expected_errors} warnings={expected_warnings}  "
              f"got errors={len(errors)} warnings={len(warnings)}")
        for i in issues:
            print(f"         {i['level'].upper()} line {i['line']}: {i['message']}")
        failures.append(name)


def write_tmp(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                    delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return Path(f.name)


print("\n=== Annotation Validator Tests ===\n")

# ── VALID CASES ──────────────────────────────────────────────────────────────
print("-- Valid inputs --")

# 1. Single rider box
p = write_tmp("0 0.500000 0.500000 0.400000 0.600000\n")
check("single rider box", validate_annotation_file(p), 0, 0)
p.unlink()

# 2. Multiple classes
p = write_tmp(
    "0 0.25 0.50 0.30 0.60\n"   # rider
    "1 0.60 0.40 0.20 0.30\n"   # with_helmet
    "2 0.80 0.70 0.15 0.20\n"   # without_helmet
    "3 0.50 0.90 0.12 0.04\n"   # number_plate
)
check("all four classes", validate_annotation_file(p), 0, 0)
p.unlink()

# 3. Comment lines and blank lines should be silently skipped
p = write_tmp(
    "# This is a comment\n"
    "\n"
    "0 0.5 0.5 0.3 0.4\n"
)
check("comment + blank lines skipped", validate_annotation_file(p), 0, 0)
p.unlink()

# 4. Empty file (no objects — valid: some frames have no targets)
p = write_tmp("")
check("empty file (no objects)", validate_annotation_file(p), 0, 0)
p.unlink()

# 5. Plate at boundary coords
p = write_tmp("3 0.999 0.999 0.001 0.001\n")
check("plate at min valid size (0.001)", validate_annotation_file(p), 0, 0)
p.unlink()

# ── INVALID CASES ────────────────────────────────────────────────────────────
print("\n-- Invalid inputs (should produce errors/warnings) --")

# 6. Invalid class ID = 5
p = write_tmp("5 0.5 0.5 0.3 0.4\n")
check("invalid class_id=5 -> 1 error", validate_annotation_file(p), 1, 0)
p.unlink()

# 7. Coordinate > 1.0
p = write_tmp("0 1.1 0.5 0.3 0.4\n")
check("x > 1.0 -> 1 error", validate_annotation_file(p), 1, 0)
p.unlink()

# 8. Negative coordinate
p = write_tmp("0 -0.1 0.5 0.3 0.4\n")
check("negative x -> 1 error", validate_annotation_file(p), 1, 0)
p.unlink()

# 9. Width below minimum (0.00001 < 0.0005)
p = write_tmp("0 0.5 0.5 0.00001 0.3\n")
check("width below MIN_BOX_SIZE -> 1 error", validate_annotation_file(p), 1, 0)
p.unlink()

# 10. Too few fields
p = write_tmp("0 0.5 0.5 0.3\n")
check("4 fields instead of 5 -> 1 error", validate_annotation_file(p), 1, 0)
p.unlink()

# 11. Non-numeric value
p = write_tmp("0 abc 0.5 0.3 0.4\n")
check("non-numeric coord -> 1 error", validate_annotation_file(p), 1, 0)
p.unlink()

# 12. Duplicate box -> warning
p = write_tmp(
    "0 0.5 0.5 0.3 0.4\n"
    "0 0.5 0.5 0.3 0.4\n"
)
check("duplicate box -> 0 errors, 1 warning", validate_annotation_file(p), 0, 1)
p.unlink()

# 13. Plate with bad aspect ratio (w=0.01, h=0.3 -> aspect=0.033 < 0.1 threshold)
p = write_tmp("3 0.5 0.5 0.01 0.30\n")
check("plate bad aspect ratio -> 0 errors, 1 warning", validate_annotation_file(p), 0, 1)
p.unlink()

# 14. Multiple errors in one file
p = write_tmp(
    "9 0.5 0.5 0.3 0.4\n"   # bad class
    "0 1.5 0.5 0.3 0.4\n"   # x out of range
    "0 0.5 0.5\n"            # too few fields
)
check("three errors in one file", validate_annotation_file(p), 3, 0)
p.unlink()

# ── DATASET-LEVEL VALIDATION ─────────────────────────────────────────────────
print("\n-- Dataset-level validation --")

import tempfile, os
tmpdir = Path(tempfile.mkdtemp())
(tmpdir / "imgdir").mkdir()
(tmpdir / "lbldir").mkdir()

# Create 2 matching image+label pairs and 1 orphan label
(tmpdir / "imgdir" / "frameA.jpg").write_bytes(b"\xff\xd8\xff\xe0")  # fake jpg
(tmpdir / "imgdir" / "frameB.jpg").write_bytes(b"\xff\xd8\xff\xe0")
(tmpdir / "lbldir" / "frameA.txt").write_text("0 0.5 0.5 0.3 0.4\n", encoding="utf-8")
(tmpdir / "lbldir" / "frameB.txt").write_text("1 0.5 0.5 0.2 0.3\n", encoding="utf-8")
(tmpdir / "lbldir" / "frameC.txt").write_text("2 0.5 0.5 0.2 0.3\n", encoding="utf-8")  # orphan

result = validate_dataset(tmpdir / "lbldir", images_dir=tmpdir / "imgdir")
ok = (result["status"] == "passed"
      and result["errors_total"] == 0
      and result["files_checked"] == 3
      and "frameC" in result["orphan_labels"]
      and "frameB" not in result["missing_labels"])
status = PASS if ok else FAIL
print(f"  [{status}] dataset: 3 labels, 2 images, 1 orphan detected")
if not ok:
    failures.append("dataset orphan detection")
    print(f"         result={result}")

# Cleanup tmpdir
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print(f"\n{'='*40}")
if failures:
    print(f"\033[31mFAILED {len(failures)} test(s):\033[0m")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"\033[32mAll tests passed.\033[0m")
    sys.exit(0)
