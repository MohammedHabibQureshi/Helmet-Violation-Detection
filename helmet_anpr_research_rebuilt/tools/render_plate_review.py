"""Create a labeled enlargement sheet for all retained plate crops."""
from pathlib import Path
import cv2
import math

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "outputs" / "review" / "label_plate_crops"
OUT = ROOT / "outputs" / "review" / "label_plate_crop_contact_sheet.jpg"
tiles = []
for path in sorted(CROPS.glob("*.jpg")):
    image = cv2.imread(str(path))
    image = cv2.resize(image, (320, 180), interpolation=cv2.INTER_CUBIC)
    canvas = cv2.copyMakeBorder(image, 28, 8, 6, 6, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(canvas, path.stem, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1, cv2.LINE_AA)
    tiles.append(canvas)
cols = 4
rows = math.ceil(len(tiles)/cols)
blank = tiles[0] * 0
while len(tiles) < rows * cols: tiles.append(blank)
sheet = cv2.vconcat([cv2.hconcat(tiles[r*cols:(r+1)*cols]) for r in range(rows)])
cv2.imwrite(str(OUT), sheet)
print(OUT)
