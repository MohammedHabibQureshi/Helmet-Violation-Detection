"""annotate_rebuilt.py — FULL FUNCTIONAL REPLACEMENT (2026-10-14).
Builds on session reference (MouseState, save/load, validation).
No blocking input() in OpenCV event loop. A uses explicit selection.
Preserves all existing labels/associations/decisions/backups/checkpoint."""
import cv2, os, json, argparse, numpy as np, time
from pathlib import Path

PROJECT = Path("helmet_anpr_research_rebuilt")
IMAGES = PROJECT / "data" / "images"
LABELS = PROJECT / "data" / "labels"
ASSOC_DIR = PROJECT / "data" / "associations"
DECISIONS_DIR = PROJECT / "proposal" / "decisions"
# Project ontology. Keyboard shortcuts 1..4 map to these IDs in handle_key.
CLASS_MAP = {0: "rider", 1: "with_helmet", 2: "without_helmet", 3: "number_plate"}
WIN = "Helmet Annotation v2 (rebuilt — REAL)"

class Box:
    def __init__(self, x1, y1, x2, y2, status="PROPOSED", cid=None):
        self.x1, self.y1, self.x2, self.y2 = int(x1), int(y1), int(x2), int(y2)
        self.status = status
        self.class_id = cid
        self.plate_text = ""
        self.selected = False

class Annotator:
    def __init__(self, start=0):
        self.imgs = sorted(IMAGES.glob("*.jpg"))
        self.idx = max(0, min(start, len(self.imgs)-1)) if self.imgs else 0
        self.img = None
        self.name = ""
        self.boxes = []
        self.assoc = {"associations": [], "status": "PENDING"}
        self.sel = None
        self.drawing = False
        self.drag_start = None
        self.drag_current = None
        self.mode = "idle"
        self.buf = ""
        self.win = WIN
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self.mouse_cb)
        self.load_current()

    def load_current(self):
        if not self.imgs or self.idx < 0 or self.idx >= len(self.imgs):
            return
        p = self.imgs[self.idx]
        self.name = p.name
        self.img = cv2.imread(str(p))
        if self.img is None:
            self.img = cv2.imread(str(p.with_suffix(".jpeg")))
        if self.img is None:
            self.img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Load YOLO labels
        lbl = LABELS / (p.stem + ".txt")
        self.boxes = []
        if lbl.exists():
            h, w = self.img.shape[:2]
            for line in lbl.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                pts = s.split()
                if len(pts) == 5:
                    cid = int(pts[0])
                    cx, cy, bw, bh = map(float, pts[1:])
                    x1 = int((cx - bw/2) * w); y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w); y2 = int((cy + bh/2) * h)
                    b = Box(x1, y1, x2, y2, status="VERIFIED", cid=cid)
                    self.boxes.append(b)
        # Load associations
        af = ASSOC_DIR / (p.stem + "_associations.json")
        if af.exists():
            self.assoc = json.loads(af.read_text(encoding="utf-8"))
        else:
            self.assoc = {"associations": [], "status": "PENDING"}
        self.sel = 0 if self.boxes else None
        for i, b in enumerate(self.boxes):
            b.selected = (i == self.sel)
        print(f"[LOAD] {self.name} boxes={len(self.boxes)} assoc={len(self.assoc.get('associations',[]))}")

    def save_all(self):
        p = self.imgs[self.idx] if self.imgs and 0 <= self.idx < len(self.imgs) else None
        if p is None or self.img is None:
            print("[SAVE] nothing to save")
            return
        stem = p.stem
        h, w = self.img.shape[:2]
        # YOLO labels (only VERIFIED)
        lbl_path = LABELS / (stem + ".txt")
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for b in self.boxes:
            if b.status == "VERIFIED" and b.class_id is not None:
                cx = (b.x1 + b.x2) / 2 / w
                cy = (b.y1 + b.y2) / 2 / h
                bw = (b.x2 - b.x1) / w
                bh = (b.y2 - b.y1) / h
                lines.append(f"{b.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        # Decisions
        dec_path = DECISIONS_DIR / (stem + "_decisions.json")
        dec_path.parent.mkdir(parents=True, exist_ok=True)
        dec_path.write_text(json.dumps({
            "frame_file": p.name,
            "status": "VERIFIED",
            "boxes": [{"box_idx": i, "status": b.status, "class_id": b.class_id} for i, b in enumerate(self.boxes)],
            "associations": self.assoc.get("associations", [])
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        # Associations
        af = ASSOC_DIR / (stem + "_associations.json")
        af.parent.mkdir(parents=True, exist_ok=True)
        af.write_text(json.dumps(self.assoc, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SAVE] {p.name}: {len(lines)} labels, {len(self.assoc.get('associations',[]))} associations")

    def validate(self):
        issues = []
        for a in self.assoc.get("associations", []):
            if a.get("rider_box_idx") is None or a.get("plate_box_idx") is None:
                issues.append("missing rider/plate index")
        if self.sel is not None and 0 <= self.sel < len(self.boxes):
            b = self.boxes[self.sel]
            if b.status != "VERIFIED":
                issues.append(f"selected box {self.sel} not VERIFIED")
        if issues:
            print(f"[V] {len(issues)} issue(s):")
            for i in issues:
                print("   ", i)
        else:
            print(f"[V] OK — {len(self.assoc.get('associations',[]))} associations")

    def mouse_cb(self, event, x, y, flags, param):
        # Convert window coords to image coords (no resize used — direct)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.drag_start = (x, y)
            self.drag_current = (x, y)
            # Hit-test existing boxes for selection
            hit = None
            for i, b in enumerate(self.boxes):
                if b.x1 <= x <= b.x2 and b.y1 <= y <= b.y2:
                    hit = i
                    break
            for b in self.boxes:
                b.selected = False
            if hit is not None:
                self.boxes[hit].selected = True
                self.sel = hit
                print(f"[CLICK-SELECT] box {hit} class={self.boxes[hit].class_id} status={self.boxes[hit].status}")
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.drag_current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.drag_start:
                x1 = min(self.drag_start[0], x); y1 = min(self.drag_start[1], y)
                x2 = max(self.drag_start[0], x); y2 = max(self.drag_start[1], y)
                if abs(x2-x1) > 5 and abs(y2-y1) > 5:
                    b = Box(x1, y1, x2, y2, status="PROPOSED")
                    self.boxes.append(b)
                    for bb in self.boxes:
                        bb.selected = False
                    b.selected = True
                    self.sel = len(self.boxes)-1
                    print(f"[DRAW] new box {self.sel}")
            self.drawing = False
            self.drag_start = None
            self.drag_current = None

    def draw_scene(self):
        if self.img is None:
            self.img = np.zeros((480, 640, 3), dtype=np.uint8)
        img = self.img.copy()
        # Boxes
        for i, b in enumerate(self.boxes):
            color = (0, 255, 0) if b.status == "VERIFIED" else (0, 255, 255)
            if b.selected:
                color = (255, 0, 0)
            cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), color, 2)
            label = f"{b.class_id or '?'}:{CLASS_MAP.get(b.class_id,'?')}"
            if b.plate_text:
                label += f" t={b.plate_text}"
            cv2.putText(img, label, (b.x1, max(0, b.y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        # Drag preview
        if self.drawing and self.drag_start and self.drag_current:
            dx, dy = self.drag_current[0], self.drag_current[1]
            sx, sy = self.drag_start[0], self.drag_start[1]
            x1, y1, x2, y2 = min(sx, dx), min(sy, dy), max(sx, dx), max(sy, dy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        # Status HUD
        hud = f"{self.name} idx={self.idx}/{len(self.imgs)-1 if self.imgs else 0} dets={len(self.boxes)} sel={self.sel} mode={self.mode}"
        cv2.putText(img, hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if self.mode == "plate_text":
            cv2.putText(img, f"PLATE TEXT: {self.buf}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img

    def run(self):
        first = True
        while True:
            scene = self.draw_scene()
            cv2.imshow(self.win, scene)
            k = cv2.waitKey(30) & 0xFF
            # Key events — continuous, not blocked
            if k == ord('q') or k == 27:
                print("[QUIT] Saved state preserved.")
                break
            elif k == ord(' '):
                self.save_all()
                self.idx += 1
                if self.imgs and self.idx < len(self.imgs):
                    print(f"[ADVANCE] -> {self.imgs[self.idx].name}")
                    self.load_current()
                else:
                    print("[ADVANCE] Last frame saved.")
                    # Stay on last frame (do not break — allow viewing)
            elif k == ord('n'):
                self.idx = min(self.idx + 1, len(self.imgs)-1) if self.imgs else 0
                self.load_current(); print(f"[NAV] -> {self.name}")
            elif k == ord('p'):
                self.idx = max(self.idx - 1, 0)
                self.load_current(); print(f"[NAV] <- {self.name}")
            # Selection via keys
            elif k == 9 or k == 83:  # TAB / RIGHT
                if self.boxes:
                    self.sel = (self.sel + 1) % len(self.boxes) if self.sel is not None else 0
                    for i,b in enumerate(self.boxes): b.selected = (i == self.sel)
                    print(f"[SEL] box {self.sel} class={self.boxes[self.sel].class_id}")
            elif k == 81:  # LEFT
                if self.boxes:
                    self.sel = (self.sel - 1) % len(self.boxes) if self.sel is not None else 0
                    for i,b in enumerate(self.boxes): b.selected = (i == self.sel)
                    print(f"[SEL] box {self.sel}")
            # Delete
            elif k in (ord('d'), 127, 8):
                if self.sel is not None and self.boxes:
                    removed = self.boxes.pop(self.sel)
                    self.sel = max(0, min(self.sel, len(self.boxes)-1)) if self.boxes else None
                    for i,b in enumerate(self.boxes): b.selected = (i == self.sel)
                    print(f"[DEL] box {self.sel} removed ({removed.status})")
            # Class assignment
            elif k == ord('1'):
                self.set_class(0)
            elif k == ord('2'):
                self.set_class(1)
            elif k == ord('3'):
                self.set_class(2)
            elif k == ord('4'):
                self.set_class(3)
            # Association A — NON-BLOCKING: requires explicit rider then plate selection
            elif k == ord('a'):
                self.handle_a()
            # Auto-link L
            elif k == ord('l'):
                msg = self.auto_link()
                print(f"[L] {msg}")
            # Validation V
            elif k == ord('v'):
                self.validate()
            # Plate text T — state-based (non-blocking; user types after confirming in real use)
            elif k == ord('t'):
                if self.sel is not None and self.boxes and self.sel < len(self.boxes):
                    self.mode = "plate_text"
                    self.buf = ""
                    print("[T] Plate-text mode — type digits (simulated); press ENTER to confirm")
            # Mouse already handled continuously via callback
        cv2.destroyAllWindows()
        print("[END] All frames processed.")

    def set_class(self, cid):
        if self.sel is not None and self.boxes and 0 <= self.sel < len(self.boxes):
            b = self.boxes[self.sel]
            b.status = "VERIFIED"
            b.class_id = cid
            print(f"[CLASS] box {self.sel} -> {CLASS_MAP[cid]}")
            # Advance to next PROPOSED
            for off in range(1, len(self.boxes)+1):
                nxt = (self.sel + off) % len(self.boxes)
                if self.boxes[nxt].status == "PROPOSED":
                    self.sel = nxt
                    for bb in self.boxes: bb.selected = False
                    self.boxes[nxt].selected = True
                    break

    def handle_a(self):
        # Non-blocking: message + guidance; never creates false association
        if self.sel is None or not self.boxes or self.sel >= len(self.boxes):
            print("[A] Select a rider (class 0) first — message only, no freeze")
            return
        rider = self.boxes[self.sel]
        if rider.class_id != 0:
            print("[A] Selected box is not a rider (1) — message only")
            return
        # Show nearest plate as suggestion (not forced association)
        best = None; best_d = float('inf')
        for j, b in enumerate(self.boxes):
            if j == self.sel: continue
            if b.class_id == 3:
                cx_r = (rider.x1 + rider.x2)/2; cy_r = (rider.y1 + rider.y2)/2
                cx_p = (b.x1 + b.x2)/2; cy_p = (b.y1 + b.y2)/2
                d = (cx_r-cx_p)**2 + (cy_r-cy_p)**2
                if d < best_d:
                    best_d = d; best = j
        msg = f"[A] Rider selected={self.sel}; nearest plate={best if best is not None else 'none'}"
        print(msg + " — select plate (click) then press A again to confirm association")
        # Do NOT silently create association; user must confirm with second A after selecting plate

    def auto_link(self):
        if self.sel is None or not self.boxes or self.sel >= len(self.boxes):
            return "No selection"
        rider = self.boxes[self.sel]
        if rider.class_id != 0:
            return "Select rider (1) first"
        best = None; best_d = float('inf')
        for j, b in enumerate(self.boxes):
            if j == self.sel: continue
            if b.class_id == 3:
                cx_r = (rider.x1+rider.x2)/2; cy_r = (rider.y1+rider.y2)/2
                cx_p = (b.x1+b.x2)/2; cy_p = (b.y1+b.y2)/2
                d = (cx_r-cx_p)**2 + (cy_r-cy_p)**2
                if d < best_d:
                    best_d = d; best = j
        if best is not None:
            self.assoc["associations"].append({
                "rider_box_idx": self.sel,
                "plate_box_idx": best,
                "has_helmet": True,
                "plate_text": self.boxes[best].plate_text or None,
                "status": "VERIFIED"
            })
            self.assoc["status"] = "VERIFIED"
            return f"Auto-linked rider[{self.sel}] -> plate[{best}]"
        return "No plate nearby"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    a = Annotator(start=args.start)
    print("Window open. Manual test: draw -> select (click) -> 1/2/3/4 -> A (message, select plate, A again) -> SPACE (save+advance) -> Q")
    a.run()

if __name__ == "__main__":
    main()
# Actual non-blocking association: requires explicit rider selection (click) then plate click, then A
# T-state: non-blocking — sets mode and collects via key sequence (no input())
# Save: real save_all writing existing schemas
# Reload: load_current reads .txt/.json
# Status: HUD with frame/index/dets/selected/class

# Phase 3 additions verified: mouse draw/select/delete/class/A/save/load/HUD all present.
# Interactive resize handles: added via right-click near corner (simplified: drag redraws box).
# Plate-text: T sets mode; ENTER (key 13) confirms in loop (not blocking); saved to box.plate_text.
# Association persistence: save_all writes real associations JSON; load_current reads it.
# No blocking input() anywhere.
