"""Pretrained YOLOv12n (no fine-tuning) on thread images and their original photos.

Run from the repo root with the project venv:
    .venv/bin/python paper/scripts/yolo_pretrained.py <instances_val2017.json> <coco_jpeg_dir> \
        <manifest.csv> <thread_image_root> <out.csv> [<coco_jpeg_dir2> ...]

<manifest.csv> needs image_id, algorithm, threads (or threads_target), source.
Thread images are found as <thread_image_root>/<image_id>.png or by the manifest
'path' column. For every image it records:
  classes found at confidence >= 0.3 (the threshold of the YOLO benchmark),
  then scores them the VLM's way (recall of the scene's COCO classes);
and, for thread images and originals separately, mAP50 against the human
COCO boxes, using the official COCO evaluator.
Writes <out.csv> (one row per image) and <out>_map50.csv (per algorithm x threads).
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
ann, jpeg_dirs, manifest, troot, out = Path(sys.argv[1]), [Path(sys.argv[2])] + [Path(p) for p in sys.argv[6:]], \
    Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
CONF = 0.3

coco = COCO(str(ann))
file2id = {im["file_name"]: im["id"] for im in coco.dataset["images"]}
name2cat = {c["name"]: c["id"] for c in coco.dataset["categories"]}
rows = list(csv.DictReader(open(manifest)))

def jpeg(src):
    for d in jpeg_dirs:
        if (d / src).exists():
            return d / src
    raise FileNotFoundError(src)

def thread_png(r):
    p = troot / f"{r['image_id']}.png"
    return p if p.exists() else troot / r["path"]

model = YOLO(str(ROOT / "models" / "yolo12n.pt"))
names = model.names

def run(paths):
    """Return per image: list of (category_id, score, [x, y, w, h])."""
    res = []
    for k in range(0, len(paths), 32):
        for p in model.predict([str(x) for x in paths[k:k + 32]], conf=0.001, iou=0.7, max_det=300,
                               device=0, half=True, verbose=False):
            dets = []
            for b, s, c in zip(p.boxes.xyxy.tolist(), p.boxes.conf.tolist(), p.boxes.cls.tolist()):
                dets.append((name2cat[names[int(c)]], s, [b[0], b[1], b[2] - b[0], b[3] - b[1]]))
            res.append(dets)
    return res

thr = run([thread_png(r) for r in rows])
org = run([jpeg(r["source"]) for r in rows])
cat2name = {v: k for k, v in name2cat.items()}

def found(dets):
    return {cat2name[c] for c, s, _ in dets if s >= CONF}

with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_id", "algorithm", "threads", "source", "truth", "found_thread", "found_original"])
    for r, dt, do in zip(rows, thr, org):
        truth = {cat2name[a["category_id"]] for a in coco.loadAnns(coco.getAnnIds(imgIds=file2id[r["source"]], iscrowd=None))}
        w.writerow([r["image_id"], r["algorithm"], r.get("threads") or r["threads_target"], r["source"],
                    "|".join(sorted(truth)), "|".join(sorted(found(dt))), "|".join(sorted(found(do)))])

def map50(img_ids, dets_by_img):
    if not img_ids:
        return 0.0
    dts = [{"image_id": i, "category_id": c, "score": s, "bbox": b} for i in img_ids for c, s, b in dets_by_img[i]]
    if not dts:
        return 0.0
    E = COCOeval(coco, coco.loadRes(dts), "bbox"); E.params.imgIds = img_ids
    E.evaluate(); E.accumulate()
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        E.summarize()
    return 100 * E.stats[1]          # AP at IoU 0.5

groups = defaultdict(list)
for k, r in enumerate(rows):
    groups[(r["algorithm"], int(r.get("threads") or r["threads_target"]))].append(k)
with open(str(out).replace(".csv", "_map50.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["algorithm", "threads", "n", "map50_thread", "map50_original", "retention"])
    for (a, t), ks in sorted(groups.items()):
        ids = [file2id[rows[k]["source"]] for k in ks]
        mt = map50(ids, {file2id[rows[k]["source"]]: thr[k] for k in ks})
        mo = map50(ids, {file2id[rows[k]["source"]]: org[k] for k in ks})
        w.writerow([a, t, len(ks), f"{mt:.1f}", f"{mo:.1f}", f"{100 * mt / mo:.1f}" if mo else "0"])
print("done", len(rows), "images")
