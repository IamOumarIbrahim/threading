"""Balanced check, step 1: pick scenes and fix their size.

Run from the repo root:
    python paper/scripts/balanced_prepare.py <coco_jpeg_dir> <work_dir>

Takes one scene from each of the 100 cells of the VLM set (IMAGE_DATASET/manifest.csv),
so the content mix matches the main study. Every scene is resized to the same pixel
area, 640 x 480 = 307,200 pixels, keeping its aspect ratio, so a given thread count
means the same thread density in every image. Writes <work_dir>/src/<scene>.png and
<work_dir>/scenes.csv (scene, classes, width, height).
"""
import csv, math, sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
coco, work = Path(sys.argv[1]), Path(sys.argv[2])
(work / "src").mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(ROOT / "IMAGE_DATASET/manifest.csv")))
cells = {}
for r in sorted(rows, key=lambda r: r["image_id"]):
    cells.setdefault((r["algorithm"], r["threads_target"]), r)
AREA = 640 * 480
with open(work / "scenes.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["scene", "classes", "width", "height"])
    for r in sorted(cells.values(), key=lambda r: r["source"]):
        im = Image.open(coco / r["source"]).convert("RGB")
        s = math.sqrt(AREA / (im.width * im.height))
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        name = r["source"].replace(".jpg", "")
        im.save(work / "src" / f"{name}.png")
        w.writerow([name, r["classes"], im.width, im.height])
print(len(cells), "scenes")
