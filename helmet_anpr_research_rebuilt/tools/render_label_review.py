"""Render current YOLO labels over all sampled source frames for visual audit."""
from pathlib import Path
import cv2
import math

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "images"
LABELS = ROOT / "data" / "labels"
OUT = ROOT / "outputs" / "review" / "label_overlay_contact_sheet.jpg"
NAMES = {0: "rider", 1: "with_helmet", 2: "without_helmet", 3: "number_plate"}
COLORS = {0: (255, 200, 0), 1: (0, 180, 0), 2: (0, 0, 255), 3: (255, 0, 255)}

def draw(image, label_path):
    h, w = image.shape[:2]
    for number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        cls, xc, yc, bw, bh = map(float, line.split())
        cls = int(cls)
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), COLORS[cls], 4)
        cv2.putText(image, f"{number}:{NAMES[cls]}", (max(2, x1), max(22, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS[cls], 2, cv2.LINE_AA)
    return image

items = []
for image_path in sorted(IMAGES.glob("*.jpg")):
    image = cv2.imread(str(image_path))
    image = draw(image, LABELS / f"{image_path.stem}.txt")
    image = cv2.resize(image, (480, 300), interpolation=cv2.INTER_AREA)
    cv2.putText(image, image_path.stem, (8, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, image_path.stem, (8, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA)
    items.append(image)

cols = 4
rows = math.ceil(len(items) / cols)
canvas = cv2.vconcat([cv2.hconcat(items[r * cols:(r + 1) * cols]) for r in range(rows)])
OUT.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT), canvas)
print(OUT)
