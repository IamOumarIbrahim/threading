"""Build the blind VLM folders for the 500 extra scenes, exactly like round one.

Run from the repo root:
    python paper/scripts/make_extra_blind.py <render_dir> <coco_jpeg_dir>

Creates
  IMAGE_DATASET_EXTRA/images/<algorithm>/<threads>/<stem>.png   (kept copies)
  IMAGE_DATASET_EXTRA/ground_truth_extra.csv                     (answers, never shown to the VLM)
  IMAGE_DATASET_EXTRA/control_map_extra.csv                      (original-photo id -> thread image id)
  ~/Desktop/Dev/blind_set_extra/          thread images named by id, shuffled order
  ~/Desktop/Dev/blind_set_extra_original/ original photos as clean PNGs under new random ids
Both blind folders get the same PROMPT.md and run.sh as round one.
"""
import csv, random, secrets, shutil, sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "IMAGE_DATASET_EXTRA"
render, jpegs = Path(sys.argv[1]), Path(sys.argv[2])
DEV = Path.home() / "Desktop" / "Dev"
BT, BO = DEV / "blind_set_extra", DEV / "blind_set_extra_original"
rows = list(csv.DictReader(open(E / "manifest_extra.csv")))

# every id already in use anywhere, so new random ids never collide
used = {r["image_id"] for r in rows}
used |= {r["identifier"] for r in csv.DictReader(open(ROOT / "IMAGE_DATASET/ground_truth.csv"))}
used |= {r["control_id"] for r in csv.DictReader(open(ROOT / "IMAGE_DATASET/control_map.csv"))}

for d in (BT / "images", BO / "images"):
    d.mkdir(parents=True, exist_ok=True)
gt_rows, map_rows = [], []
for r in rows:
    stem = Path(r["source"]).stem
    png = next((render / r["image_id"]).glob("*.png"))
    keep = E / "images" / r["algorithm"] / r["threads"] / f"{stem}.png"
    keep.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(png, keep)
    shutil.copyfile(png, BT / "images" / f"{r['image_id']}.png")
    gt_rows.append([r["image_id"], r["algorithm"], r["threads"], r["classes"], r["source"],
                    str(keep.relative_to(E))])
    while (cid := secrets.token_hex(6)) in used:
        pass
    used.add(cid)
    im = Image.open(jpegs / r["source"]); im.load()
    clean = Image.new("RGB", im.size); clean.paste(im.convert("RGB"))   # drops all metadata
    clean.save(BO / "images" / f"{cid}.png")
    map_rows.append([cid, r["source"], r["image_id"]])

with open(E / "ground_truth_extra.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["identifier", "method", "threads", "ground_truth", "source_image", "path"]); w.writerows(gt_rows)
with open(E / "control_map_extra.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["control_id", "source_image", "threaded_identifier"]); w.writerows(map_rows)

for folder, ids, seed in ((BT, [g[0] for g in gt_rows], 21), (BO, [m[0] for m in map_rows], 22)):
    random.Random(seed).shuffle(ids)
    (folder / "order.txt").write_text("\n".join(ids) + "\n")
    for f in ("PROMPT.md", "run.sh"):
        shutil.copy2(ROOT / "paper" / "vlm" / f, folder / f)
print(len(gt_rows), "thread images |", len(map_rows), "original photos")
