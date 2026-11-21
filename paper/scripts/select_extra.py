"""Pick 500 extra COCO val2017 scenes for the VLM, 10 per cell, over the
thread counts both models share (2,000 to 10,000 in steps of 2,000).

Run from the repo root:
    python paper/scripts/select_extra.py <instances_val2017.json> <out_dir>

Rules match recognition/build_dataset.py: a scene is never reused, scenes in
IMAGE_DATASET/manifest.csv or excluded_sources.txt are skipped, and scenes are
sorted by content (person or not, most common class, object count) and dealt
to shuffled cells, so every cell gets a similar mix. Scenes with no labelled
object are skipped, since they cannot contribute to recall.
Writes <out_dir>/manifest_extra.csv.
"""
import csv, json, random, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALGOS = ["anchor_web", "edge_tangent", "flow_streamlines", "greedy_error", "greedy_residual",
         "quadtree_hatch", "radon_peel", "scanline_dashes", "stipple_tour", "thread_painter"]
THREADS = [2000, 4000, 6000, 8000, 10000]
PER_CELL, SEED = 10, 20261121

ann, out = Path(sys.argv[1]), Path(sys.argv[2])
coco = json.loads(ann.read_text())
names = {c["id"]: c["name"] for c in coco["categories"]}
scenes = {i["id"]: {"file": i["file_name"], "w": i["width"], "h": i["height"], "counts": Counter()} for i in coco["images"]}
for a in coco["annotations"]:
    scenes[a["image_id"]]["counts"][names[a["category_id"]]] += 1

used = {r["source"] for r in csv.DictReader(open(ROOT / "IMAGE_DATASET/manifest.csv"))}
used |= set((ROOT / "IMAGE_DATASET/excluded_sources.txt").read_text().split())
pool = [s for s in scenes.values() if s["file"] not in used and s["counts"]]

def stratum(s):
    c = s["counts"]; top = min(c, key=lambda k: (-c[k], k))
    return ("person" in c, top, sum(c.values()))

cells = [(a, t) for a in ALGOS for t in THREADS]
need = PER_CELL * len(cells)
pool.sort(key=lambda s: (stratum(s), s["file"]))
step = len(pool) / need
picked = [pool[int((k + 0.5) * step)] for k in range(need)]

rng = random.Random(SEED)
rows, ids = [], set()
for start in range(0, need, len(cells)):
    for s, (a, t) in zip(picked[start:start + len(cells)], rng.sample(cells, len(cells))):
        while (i := f"{rng.getrandbits(48):012x}") in ids:
            pass
        ids.add(i)
        rows.append({"image_id": i, "algorithm": a, "threads": t, "source": s["file"],
                     "width": s["w"], "height": s["h"], "classes": "|".join(sorted(s["counts"])),
                     "n_objects": sum(s["counts"].values())})
out.mkdir(parents=True, exist_ok=True)
with open(out / "manifest_extra.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
per = Counter((r["algorithm"], r["threads"]) for r in rows)
print(len(rows), "scenes |", len(per), "cells |", min(per.values()), "-", max(per.values()), "per cell |",
      len({r["source"] for r in rows}), "unique sources")
