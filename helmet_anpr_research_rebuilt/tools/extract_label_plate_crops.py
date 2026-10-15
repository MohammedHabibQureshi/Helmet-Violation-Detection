"""Extract review crops strictly from the current class-3 YOLO labels."""
from pathlib import Path
import csv
import cv2

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "images"
LABELS = ROOT / "data" / "labels"
OUT = ROOT / "outputs" / "review" / "label_plate_crops"
MANIFEST = ROOT / "reports" / "LABEL_PLATE_CROP_MANIFEST.csv"
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for image_path in sorted(IMAGES.glob("*.jpg")):
    image = cv2.imread(str(image_path)); h, w = image.shape[:2]
    for idx, line in enumerate((LABELS / f"{image_path.stem}.txt").read_text(encoding="utf-8").splitlines()):
        if not line.strip(): continue
        cls, xc, yc, bw, bh = map(float, line.split())
        if int(cls) != 3: continue
        x1, y1 = max(0, int((xc-bw/2)*w)), max(0, int((yc-bh/2)*h))
        x2, y2 = min(w, int((xc+bw/2)*w)), min(h, int((yc+bh/2)*h))
        name = f"{image_path.stem}_labelidx_{idx}.jpg"
        cv2.imwrite(str(OUT/name), image[y1:y2, x1:x2])
        rows.append({"frame": image_path.stem, "label_index":idx, "crop": str((OUT/name).relative_to(ROOT)).replace('\\','/'), "x1":x1,"y1":y1,"x2":x2,"y2":y2})
with MANIFEST.open("w", newline="", encoding="utf-8") as f:
    writer=csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
print(f"{len(rows)} class-3 crops extracted")
