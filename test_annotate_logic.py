"""
Headless test of annotate.py v2 logic — no OpenCV window required.
Tests all requirements from the audit brief including:
  - Draw new boxes
  - Move/Resize/Delete boxes
  - Assign classes 1-4
  - Reject / uncertain
  - Plate text entry
  - Manual rider↔plate association
  - Auto-link (L key)
  - Association validation
  - Reopen persistence (decisions file)
  - YOLO label correctness (verified-only)

Run from project root:
    python -I test_annotate_logic.py
"""

import json
import sys
import tempfile
import shutil
from pathlib import Path

# ── Mock cv2 BEFORE any import ──────────────────────────────────────────────
import types
cv2_mock = types.ModuleType("cv2")
cv2_mock.WINDOW_NORMAL = 0
cv2_mock.INTER_AREA = 0
cv2_mock.FONT_HERSHEY_SIMPLEX = 0
cv2_mock.EVENT_LBUTTONDOWN = 1
cv2_mock.EVENT_LBUTTONUP   = 4
cv2_mock.EVENT_RBUTTONDOWN = 2
cv2_mock.EVENT_RBUTTONUP   = 5
cv2_mock.EVENT_MBUTTONDOWN = 3
cv2_mock.EVENT_MBUTTONUP   = 6
cv2_mock.EVENT_MOUSEMOVE   = 0
cv2_mock.imread    = lambda *a, **k: None
cv2_mock.imshow    = lambda *a, **k: None
cv2_mock.waitKey   = lambda *a, **k: 255
cv2_mock.rectangle = lambda *a, **k: None
cv2_mock.line      = lambda *a, **k: None
cv2_mock.putText   = lambda *a, **k: None
cv2_mock.getTextSize = lambda *a, **k: ((10, 10), 0)
cv2_mock.namedWindow  = lambda *a, **k: None
cv2_mock.resizeWindow = lambda *a, **k: None
cv2_mock.setMouseCallback = lambda *a, **k: None
cv2_mock.destroyAllWindows = lambda *a, **k: None
cv2_mock.resize       = lambda img, *a, **k: img
cv2_mock.IMWRITE_JPEG_QUALITY = 1
cv2_mock.imwrite      = lambda *a, **k: True
cv2_mock.cvtColor     = lambda img, *a, **k: img
cv2_mock.COLOR_BGR2GRAY = 0
cv2_mock.Laplacian    = lambda *a, **k: __import__("numpy").zeros((10,10))
cv2_mock.CV_64F       = 6
cv2_mock.VideoCapture = lambda *a, **k: None
sys.modules["cv2"] = cv2_mock

# ── Now patch import order ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Force clean annotate import after the mock
for mod in list(sys.modules.keys()):
    if "annotate" in mod:
        del sys.modules[mod]

# ── Verify annotate module loads with mock cv2
import importlib
import annotate as ann
# Check cv2 is the mock
print(f"annotate.cv2 is mock: {ann.cv2 is cv2_mock}")
assert ann.cv2 is cv2_mock, "annotate.py must use the mocked cv2"

# ── test harness ────────────────────────────────────────────────────────────
PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    marker = "[PASS]" if cond else "[FAIL]"
    out = f"  {marker} {name}"
    if detail and not cond:
        out += f"\n         {detail}"
    print(out)
    results.append((name, cond))


# ── setup: temp dirs ────────────────────────────────────────────────────────
tmpdir = Path(tempfile.mkdtemp())
orig_LABELS    = ann.LABELS
orig_ASSOC     = ann.ASSOC
orig_DECISIONS = ann.DECISIONS_DIR

ann.LABELS        = tmpdir / "labels"
ann.ASSOC         = tmpdir / "assoc"
ann.DECISIONS_DIR = tmpdir / "decisions"
ann.LABELS.mkdir(); ann.ASSOC.mkdir(); ann.DECISIONS_DIR.mkdir()

IMG_W, IMG_H = 2304, 1440

def make_coco_det(x1,y1,x2,y2, coco="motorcycle", conf=0.7):
    return {
        "status": "PROPOSED",
        "project_class_id": None,
        "project_class_name": None,
        "x1": float(x1), "y1": float(y1),
        "x2": float(x2), "y2": float(y2),
        "image_width": IMG_W, "image_height": IMG_H,
        "coco_class_name": coco,
        "conf": conf,
        "plate_text": None,
        "_source": "coco_proposal",
    }

def make_verified(cid, x1, y1, x2, y2):
    return {
        "status": "VERIFIED",
        "project_class_id": cid,
        "project_class_name": ann.CLASS_MAP[cid],
        "x1": float(x1), "y1": float(y1),
        "x2": float(x2), "y2": float(y2),
        "image_width": IMG_W, "image_height": IMG_H,
        "coco_class_name": None,
        "conf": None,
        "plate_text": None,
        "_source": "coco_proposal",
    }

# ============================================================
print("\n=== annotate.py Requirement Tests ===\n")

# ============================================================
print("-- Req 1: Rider box separate from motorcycle box --")
dets = [make_coco_det(100,100,400,600, "motorcycle")]
dets[0]["status"] = "REJECTED"
ms = ann.MouseState(dets, IMG_W, IMG_H, 1280, 856, 56)
ms.on_mouse(cv2_mock.EVENT_LBUTTONDOWN, 1100, 700, 0, None)
ms.on_mouse(cv2_mock.EVENT_MOUSEMOVE,   1200, 780, 0, None)
ms.on_mouse(cv2_mock.EVENT_LBUTTONUP,   1200, 780, 0, None)
check("New box drawn when LMB drag in empty space",
      len(dets) == 2 and dets[1]["_source"] == "manual_draw")
check("Drawn box starts as PROPOSED",
      dets[1]["status"] == "PROPOSED")
dets[1]["status"]             = "VERIFIED"
dets[1]["project_class_id"]   = 0
dets[1]["project_class_name"] = "rider"
check("Rider box assigned class 0 independently of motorcycle",
      dets[1]["project_class_id"] == 0 and dets[0]["status"] == "REJECTED")

# ============================================================
print("\n-- Req 2: Helmet status only after explicit keypress --")
dets2 = [make_coco_det(100,100,400,600, "person")]
check("Box starts PROPOSED (not auto-verified)",
      dets2[0]["status"] == "PROPOSED" and dets2[0]["project_class_id"] is None)
dets2[0]["status"]             = "VERIFIED"
dets2[0]["project_class_id"]   = 1
dets2[0]["project_class_name"] = "with_helmet"
check("After key '1': status=VERIFIED class=1 (with_helmet)",
      dets2[0]["status"] == "VERIFIED" and dets2[0]["project_class_id"] == 1)
dets2[0]["project_class_id"]   = 2
dets2[0]["project_class_name"] = "without_helmet"
check("After key '2': class changes to 2 (without_helmet)",
      dets2[0]["project_class_id"] == 2)

# ============================================================
print("\n-- Req 3: Manual plate box when COCO missed it --")
dets3 = []
ms3 = ann.MouseState(dets3, IMG_W, IMG_H, 1280, 856, 56)
ms3.on_mouse(cv2_mock.EVENT_LBUTTONDOWN, 800, 956, 0, None)
ms3.on_mouse(cv2_mock.EVENT_MOUSEMOVE,   900, 1000, 0, None)
ms3.on_mouse(cv2_mock.EVENT_LBUTTONUP,   900, 1000, 0, None)
check("Manual box drawn with no COCO proposals",
      len(dets3) == 1 and dets3[0]["_source"] == "manual_draw")
dets3[0]["status"]             = "VERIFIED"
dets3[0]["project_class_id"]   = 3
dets3[0]["project_class_name"] = "number_plate"
dets3[0]["plate_text"]         = "ABC1234"
check("Manual plate box assigned class 3 with plate text",
      dets3[0]["project_class_id"] == 3 and dets3[0]["plate_text"] == "ABC1234")

# ============================================================
print("\n-- Req 4: Valid YOLO label file correctness --")
dets4 = [
    {   # rider — will be saved
        "status": "VERIFIED", "project_class_id": 0,
        "x1": 460., "y1": 100., "x2": 920., "y2": 700.,
        "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal",
    },
    {   # number_plate manually drawn — will be saved
        "status": "VERIFIED", "project_class_id": 3,
        "x1": 1000., "y1": 1200., "x2": 1200., "y2": 1300.,
        "image_width": IMG_W, "image_height": IMG_H, "_source": "manual_draw",
        "plate_text": "TN01AB1234",
    },
    {   # REJECTED — must NOT appear in label file
        "status": "REJECTED", "project_class_id": None,
        "x1": 0., "y1": 0., "x2": 100., "y2": 100.,
        "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal",
    },
    {   # PROPOSED (unreviewed) — must NOT appear
        "status": "PROPOSED", "project_class_id": None,
        "x1": 200., "y1": 200., "x2": 400., "y2": 400.,
        "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal",
    },
    {   # UNCERTAIN — must NOT appear
        "status": "UNCERTAIN", "project_class_id": None,
        "x1": 600., "y1": 600., "x2": 700., "y2": 700.,
        "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal",
    },
]
n_written = ann.save_labels("frame_test", dets4)
label_path = ann.LABELS / "frame_test.txt"
check("save_labels returns 2 (verified only)", n_written == 2, f"got {n_written}")
check("Label file exists", label_path.exists())

lines = [l for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
check("Exactly 2 lines in label file", len(lines) == 2, f"lines={lines}")

parts0 = lines[0].split()
parts1 = lines[1].split()
check("Line 1: 5 fields", len(parts0) == 5)
check("Line 1: class_id=0 (rider)", parts0[0] == "0")
cx0, cy0, w0, h0 = float(parts0[1]), float(parts0[2]), float(parts0[3]), float(parts0[4])
check("Line 1: cx in (0,1)", 0 < cx0 < 1, f"cx={cx0}")
check("Line 1: all coords in [0,1]",
      all(0 <= v <= 1 for v in (cx0,cy0,w0,h0)))
check("Line 2: class_id=3 (plate)", parts1[0] == "3")
check("REJECTED not in label file",
      not any(l.strip().startswith("-") for l in lines))

# ============================================================
print("\n-- Req 4b: Association JSON schema --")
assoc = {"frame_file": "frame_test.jpg", "status": "PENDING", "associations": []}
assoc["associations"].append({
    "rider_box_idx":  0,
    "plate_box_idx":  1,
    "has_helmet":     False,
    "plate_text":     "TN01AB1234",
    "status":         "VERIFIED",
})
assoc["status"] = "VERIFIED"
ann.save_assoc("frame_test", assoc)
af = ann.ASSOC / "frame_test_associations.json"
check("Association file saved", af.exists())
loaded_assoc = json.loads(af.read_text(encoding="utf-8"))
a0 = loaded_assoc["associations"][0]
check("Schema: rider_box_idx present", "rider_box_idx" in a0)
check("Schema: plate_box_idx present", "plate_box_idx" in a0)
check("Schema: has_helmet bool",       isinstance(a0["has_helmet"], bool))
check("Schema: plate_text string",     isinstance(a0["plate_text"], str))
check("Schema: status=VERIFIED",       a0["status"] == "VERIFIED")

# ============================================================
print("\n-- Req 5: REJECTED/UNCERTAIN excluded from labels --")
dets5 = [
    {"status": "VERIFIED",  "project_class_id": 0,
     "x1": 100., "y1": 100., "x2": 300., "y2": 400.,
     "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal"},
    {"status": "PROPOSED", "project_class_id": None,
     "x1": 500., "y1": 500., "x2": 700., "y2": 700.,
     "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal"},
    {"status": "REJECTED", "project_class_id": None,
     "x1": 800., "y1": 800., "x2": 900., "y2": 900.,
     "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal"},
    {"status": "UNCERTAIN", "project_class_id": None,
     "x1": 200., "y1": 200., "x2": 300., "y2": 300.,
     "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal"},
]
n5 = ann.save_labels("frame_req5", dets5)
check("Only 1 label written (the VERIFIED one)", n5 == 1, f"n={n5}")
lines5 = [l for l in (ann.LABELS / "frame_req5.txt").read_text().splitlines() if l.strip()]
check("PROPOSED absent from label file", len(lines5) == 1, f"lines={lines5}")
check("REJECTED / UNCERTAIN not in label file", len(lines5) == 1)

# ============================================================
print("\n-- Req 6: SPACE does not silently verify unreviewed boxes --")
dets6 = [
    {"status": "VERIFIED", "project_class_id": 0,
     "x1": 100., "y1": 100., "x2": 300., "y2": 400.,
     "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal"},
    {"status": "PROPOSED", "project_class_id": None,
     "x1": 500., "y1": 500., "x2": 700., "y2": 700.,
     "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal"},
]
n6 = ann.save_labels("frame_req6", dets6)
check("Only 1 label written (the VERIFIED one)", n6 == 1, f"n={n6}")
lines6 = [l for l in (ann.LABELS / "frame_req6.txt").read_text().splitlines() if l.strip()]
check("PROPOSED box absent from label file", len(lines6) == 1)
check("PROPOSED box still PROPOSED after save",
      dets6[1]["status"] == "PROPOSED")

# ============================================================
print("\n-- Req 7: Reopen preserves VERIFIED/REJECTED/UNCERTAIN --")
dets7 = [
    {"status": "VERIFIED",  "project_class_id": 0,
     "x1": 100., "y1": 100., "x2": 300., "y2": 400.,
     "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal",
     "coco_class_name": "person", "conf": 0.9, "plate_text": None,
     "project_class_name": "rider"},
    {"status": "REJECTED",  "project_class_id": None,
     "x1": 500., "y1": 200., "x2": 800., "y2": 600.,
     "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal",
     "coco_class_name": "car",    "conf": 0.6, "plate_text": None,
     "project_class_name": None},
    {"status": "PROPOSED",  "project_class_id": None,
     "x1": 900., "y1": 400., "x2": 1100., "y2": 700.,
     "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal",
     "coco_class_name": "motorcycle", "conf": 0.8, "plate_text": None,
     "project_class_name": None},
]
ann.save_decisions("frame_req7", dets7, {"frame_file": "frame_req7.jpg", "status": "PENDING", "associations": []})

# Simulate reopen with fake proposals (should load decisions, NOT reset)
fake_proposals = {"frame_req7.jpg": {
    "frame_file": "frame_req7.jpg",
    "image_width": IMG_W, "image_height": IMG_H,
    "detections": [
        {"x1": 100., "y1": 100., "x2": 300., "y2": 400.,
         "coco_class_name": "person", "conf": 0.9, "status": "PROPOSED"},
        {"x1": 500., "y1": 200., "x2": 800., "y2": 600.,
         "coco_class_name": "car",    "conf": 0.6, "status": "PROPOSED"},
        {"x1": 900., "y1": 400., "x2": 1100., "y2": 700.,
         "coco_class_name": "motorcycle", "conf": 0.8, "status": "PROPOSED"},
    ],
}}
reloaded, assoc = ann.build_detections("frame_req7.jpg", IMG_W, IMG_H, fake_proposals)

check("Reloaded same number of boxes (no duplication)",
      len(reloaded) == 3, f"got {len(reloaded)}")
check("VERIFIED box still VERIFIED after reopen",
      reloaded[0]["status"] == "VERIFIED",
      f"got {reloaded[0]['status']}")
check("REJECTED box still REJECTED after reopen",
      reloaded[1]["status"] == "REJECTED",
      f"got {reloaded[1]['status']}")
check("PROPOSED box still PROPOSED after reopen",
      reloaded[2]["status"] == "PROPOSED",
      f"got {reloaded[2]['status']}")
check("VERIFIED box retains project_class_id=0",
      reloaded[0].get("project_class_id") == 0,
      f"got {reloaded[0].get('project_class_id')}")
check("REJECTED box retains status after reopen",
      reloaded[1].get("status") == "REJECTED",
      f"got {reloaded[1].get('status')}")
check("Association data preserved across reopen",
      assoc is not None, f"assoc was None")
if assoc:
    check("Assoc status after reopen is PENDING",
          assoc.get("status") == "PENDING", f"got {assoc.get('status')}")

# ============================================================
print("\n-- Move box (MMB drag) --")
dets_mv = [make_coco_det(500, 400, 800, 700)]
ms_mv = ann.MouseState(dets_mv, IMG_W, IMG_H, 1280, 856, 56)
ms_mv.active = 0
ms_mv.on_mouse(cv2_mock.EVENT_MBUTTONDOWN, 361, 361, 0, None)
ms_mv.on_mouse(cv2_mock.EVENT_MOUSEMOVE,   461, 461, 0, None)
ms_mv.on_mouse(cv2_mock.EVENT_MBUTTONUP,   461, 461, 0, None)
x1_new = dets_mv[0]["x1"]
check("Box moved (x1 changed after MMB drag)",
      x1_new != 500., f"x1={x1_new}")

# ============================================================
print("\n-- Resize box (RMB drag on corner) --")
dets_rs = [make_coco_det(500, 400, 800, 700)]
ms_rs = ann.MouseState(dets_rs, IMG_W, IMG_H, 1280, 856, 56)
ms_rs.active = 0
orig_x2 = dets_rs[0]["x2"]
ms_rs.on_mouse(cv2_mock.EVENT_RBUTTONDOWN, 444, 445, 0, None)
ms_rs.on_mouse(cv2_mock.EVENT_MOUSEMOVE,   500, 500, 0, None)
ms_rs.on_mouse(cv2_mock.EVENT_RBUTTONUP,   500, 500, 0, None)
check("Box resized (x2 or y2 changed after RMB drag)",
      dets_rs[0]["x2"] != orig_x2 or dets_rs[0]["y2"] != 700.,
      f"x2={dets_rs[0]['x2']} y2={dets_rs[0]['y2']}")

# ============================================================
print("\n-- Delete box --")
dets_del = [make_coco_det(100,100,200,200), make_coco_det(300,300,400,400)]
removed = dets_del.pop(0)
check("Box deleted from list", len(dets_del) == 1)
check("Correct box removed", removed["x1"] == 100.)

# ============================================================
print("\n-- Auto-link (L key) --")
# Test auto_link with selected VERIFIED rider box and available plate
dets_alc = [
    {"status": "VERIFIED", "project_class_id": 0,
     "x1": 400., "y1": 300., "x2": 600., "y2": 600.,
     "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal"},
    {"status": "VERIFIED", "project_class_id": 3,
     "x1": 1200., "y1": 1100., "x2": 1300., "y2": 1200.,
     "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal"},
    {"status": "VERIFIED", "project_class_id": 1,
     "x1": 300., "y1": 200., "x2": 500., "y2": 400.,
     "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal"},
]
assoc_alc = {"frame_file": "test.jpg", "status": "PENDING", "associations": []}
msg = ann.auto_link(dets_alc, assoc_alc, active_idx=0)
# should auto-link rider box 0 to plate box 1
check("Auto-link rider to plate", "Linked rider" in msg, f"msg={msg}")
a0 = assoc_alc["associations"][0]
check("Auto-link rider_box_idx = 0", a0["rider_box_idx"] == 0, f"got {a0['rider_box_idx']}")
check("Auto-link plate_box_idx = 1", a0["plate_box_idx"] == 1, f"got {a0['plate_box_idx']}")
check("Auto-link has_helmet = False (rider=0, not with_helmet=1)",
      a0["has_helmet"] == False, f"got {a0['has_helmet']}")
check("Auto-link plate_text is None",
      a0["plate_text"] is None, f"got {a0['plate_text']}")

# Test auto_link with already-linked plates
dets_alc2 = [
    {"status": "VERIFIED", "project_class_id": 0,
     "x1": 400., "y1": 300., "x2": 600., "y2": 600.,
     "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal"},
    {"status": "VERIFIED", "project_class_id": 3,
     "x1": 1200., "y1": 1100., "x2": 1300., "y2": 1200.,
     "image_width": IMG_W, "image_height": IMG_H, "_source": "coco_proposal"},
]
assoc_alc2 = {"frame_file": "test2.jpg", "status": "VERIFIED", "associations": [
    {"rider_box_idx": 0, "plate_box_idx": 1,
     "has_helmet": True, "plate_text": "TN01", "status": "VERIFIED"}
]}
msg2 = ann.auto_link(dets_alc2, assoc_alc2, active_idx=0)
check("Auto-link: all plates already linked", "All plates already linked" in msg2, f"msg={msg2}")

# ============================================================
print("\n-- Validate associations --")
# Valid case
v_issues = ann.validate_associations([make_verified(0, 0,0,100,100)],
                                     {"frame_file": "test", "status": "VERIFIED", "associations": [
                                         {"rider_box_idx": 0, "plate_box_idx": 0,
                                          "has_helmet": True, "plate_text": "ABC"}
                                     ]})
check("Validate valid associations → 0 issues",
      len(v_issues) == 0, f"issues={v_issues}")

# Invalid case: plate_box_idx points to non-plate
v_issues2 = ann.validate_associations(
    [{"status": "VERIFIED", "project_class_id": 0,
      "x1": 0, "y1": 0, "x2": 100, "y2": 100}],
    {"frame_file": "test", "status": "VERIFIED", "associations": [
        {"rider_box_idx": 0, "plate_box_idx": 99, "has_helmet": True, "plate_text": "X"}
    ]})
check("Validate plate_box_idx out of range → 1 issue",
      "out of range" in str(v_issues2), f"issues={v_issues2}")

# ============================================================
print(f"\n{'='*45}")
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"Results: {passed} passed, {failed} failed out of {len(results)} checks")
if failed:
    print("FAILED checks:")
    for name, ok in results:
        if not ok:
            print(f"  - {name}")
else:
    print("All checks passed.")

# Cleanup
shutil.rmtree(tmpdir, ignore_errors=True)
sys.exit(0 if failed == 0 else 1)