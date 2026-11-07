#!/usr/bin/env python3
"""Render the human recognition dataset: 10 algorithms x 20 budgets x 25 images.

Every COCO val2017 image is used exactly once, so no rater sees a scene twice.
Each cell receives the same 15/5/5 mix of the benchmark's train/val/test
splits, and scenes of similar content are dealt evenly across cells.

Usage (from the project root):
    python recognition/build_dataset.py --out D:/IMAGE_DATASET
    python recognition/build_dataset.py --out D:/IMAGE_DATASET --verify

Re-running skips images that already exist. The same seeds give the same
assignment and byte-identical images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ALGORITHMS = {
    "greedy_residual": 1, "flow_streamlines": 2, "scanline_dashes": 3,
    "radon_peel": 4, "stipple_tour": 5, "edge_tangent": 6, "greedy_error": 7,
    "quadtree_hatch": 8, "anchor_web": 9, "thread_painter": 10,
}
THREAD_COUNTS = list(range(1000, 20001, 1000))
PER_CELL = {"train": 15, "val": 5, "test": 5}
ASSIGN_SEED = 20261107   # scene -> cell assignment
RENDER_SEED = 7          # threader default, also used for the YOLO benchmark
SPLIT_SEED = 1           # same 60/20/20 split as run_threaded_experiment.py

FIELDS = ["image_id", "path", "algorithm", "algorithm_id", "threads_target",
          "threads_actual", "source", "split", "width", "height", "classes",
          "n_objects", "render_seed", "bytes", "sha256"]


def load_scenes(annotations: Path) -> dict:
    """Return {file_name: scene} with size and per-class object counts."""
    coco = json.loads(annotations.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    scenes = {i["id"]: {"file_name": i["file_name"], "width": i["width"],
                        "height": i["height"], "counts": Counter()}
              for i in coco["images"]}
    for ann in coco["annotations"]:
        scenes[ann["image_id"]]["counts"][names[ann["category_id"]]] += 1
    return {s["file_name"]: s for s in scenes.values()}


def make_splits(names: list[str]) -> dict[str, list[str]]:
    """Reproduce the benchmark's seed-1 60/20/20 split."""
    shuffled = sorted(names)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    n_train, n_val = int(len(shuffled) * 0.6), int(len(shuffled) * 0.2)
    return {"train": shuffled[:n_train],
            "val": shuffled[n_train:n_train + n_val],
            "test": shuffled[n_train + n_val:]}


def assign(scenes: dict) -> list[dict]:
    """Deal every scene to exactly one algorithm x budget cell."""
    cells = [(a, t) for a in ALGORITHMS for t in THREAD_COUNTS]
    expected = {k: v * len(cells) for k, v in PER_CELL.items()}
    splits = make_splits(list(scenes))
    got = {k: len(v) for k, v in splits.items()}
    if got != expected:
        sys.exit(f"Expected splits {expected}, found {got}")

    rng = random.Random(ASSIGN_SEED)
    items = []
    for split, names in splits.items():
        def stratum(name):
            counts = scenes[name]["counts"]
            top = min(counts, key=lambda c: (-counts[c], c)) if counts else ""
            return "person" in counts, top, sum(counts.values())

        # Sorted scenes dealt to shuffled cells: every cell gets one scene from
        # each run of similar scenes, so content stays balanced across cells.
        ordered = sorted(sorted(names), key=stratum)
        for start in range(0, len(ordered), len(cells)):
            round_cells = rng.sample(cells, len(cells))
            for name, (algo, threads) in zip(ordered[start:], round_cells):
                counts = scenes[name]["counts"]
                items.append({
                    "image_id": f"{rng.getrandbits(48):012x}",
                    "algorithm": algo,
                    "algorithm_id": ALGORITHMS[algo],
                    "threads_target": threads,
                    "source": name,
                    "split": split,
                    "width": scenes[name]["width"],
                    "height": scenes[name]["height"],
                    "classes": "|".join(sorted(counts)),
                    "n_objects": sum(counts.values()),
                })
    if len({i["image_id"] for i in items}) != len(items):
        sys.exit("Image id collision; change ASSIGN_SEED")
    return sorted(items, key=lambda i: (i["algorithm_id"], i["threads_target"],
                                        i["source"]))


def image_path(item: dict) -> str:
    stem = Path(item["source"]).stem
    return f"images/{item['algorithm']}/{item['threads_target']}/{stem}.png"


def render(item: dict, exe: Path, dataset: Path, out: Path) -> dict:
    """Render one item and complete its manifest row. Skips finished images."""
    final = out / item["path"]
    count_file = final.with_suffix(".count")
    if final.exists() and count_file.exists():
        item["threads_actual"] = int(count_file.read_text())
    else:
        algo = item["algorithm"]
        with tempfile.TemporaryDirectory(dir=out / ".tmp") as tmp:
            proc = subprocess.run(
                [str(exe), str(dataset / item["source"]),
                 "--only", str(item["algorithm_id"]),
                 "--threads", str(item["threads_target"]),
                 "--seed", str(RENDER_SEED), "--out", tmp],
                capture_output=True, text=True, timeout=900)
            produced = Path(tmp) / f"{Path(item['source']).stem}_{algo}.png"
            match = re.search(rf"{algo}\s+(\d+) threads", proc.stdout)
            if proc.returncode or not produced.exists() or not match:
                raise RuntimeError(f"{item['source']} {algo}: {proc.stderr[-300:]}")
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(produced, final)
        item["threads_actual"] = int(match.group(1))
        count_file.write_text(str(item["threads_actual"]))
    data = final.read_bytes()
    item["bytes"] = len(data)
    item["sha256"] = hashlib.sha256(data).hexdigest()
    item["render_seed"] = RENDER_SEED
    return item


def write_manifest(items: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(sorted(items, key=lambda i: i["path"]))
    os.replace(tmp, path)


def verify(out: Path) -> int:
    with open(out / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    problems = []
    cells = Counter((r["algorithm"], r["threads_target"]) for r in rows)
    n_cells = len(ALGORITHMS) * len(THREAD_COUNTS)
    if len(cells) != n_cells or set(cells.values()) != {sum(PER_CELL.values())}:
        problems.append(f"unequal cells: {sorted(Counter(cells.values()).items())}")
    if len({r["source"] for r in rows}) != len(rows):
        problems.append("repeated source scene")
    for r in rows:
        p = out / r["path"]
        if not p.exists():
            problems.append(f"missing {r['path']}")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != r["sha256"]:
            problems.append(f"hash mismatch {r['path']}")
    print(f"{len(rows)} images, {len(cells)} cells, {len(problems)} problems")
    for p in problems[:20]:
        print(" ", p)
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("D:/IMAGE_DATASET"))
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    ap.add_argument("--exe", type=Path, default=ROOT / "build" / "threader.exe")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, help="render only the first N (smoke test)")
    ap.add_argument("--verify", action="store_true", help="check an existing dataset")
    args = ap.parse_args()

    if args.verify:
        return verify(args.out)

    for need in (args.exe, args.dataset / "annotations.json"):
        if not need.exists():
            sys.exit(f"Missing {need}")

    items = assign(load_scenes(args.dataset / "annotations.json"))
    for item in items:
        item["path"] = image_path(item)
    if args.limit:
        items = random.Random(0).sample(items, args.limit)

    (args.out / ".tmp").mkdir(parents=True, exist_ok=True)
    start, done, failed = time.time(), [], []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(render, i, args.exe, args.dataset, args.out)
                   for i in items]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                done.append(fut.result())
            except Exception as exc:  # keep going; a rerun retries failures
                failed.append(str(exc))
                print(f"FAILED {exc}", flush=True)
            if n % 100 == 0 or n == len(items):
                rate = n / (time.time() - start)
                print(f"{n}/{len(items)}  {rate:.1f}/s  "
                      f"{(len(items) - n) / rate / 60:.1f} min left", flush=True)

    shutil.rmtree(args.out / ".tmp", ignore_errors=True)
    if failed:
        print(f"{len(failed)} failed; run again to retry.")
        return 1
    if args.limit:
        print(f"Smoke test OK: {len(done)} images (no manifest written).")
        return 0

    write_manifest(done, args.out / "manifest.csv")
    for p in (args.out / "images").rglob("*.count"):
        p.unlink()

    ratios = defaultdict(list)
    for i in done:
        ratios[(i["algorithm"], i["threads_target"])].append(
            i["threads_actual"] / i["threads_target"])
    worst = sorted((sum(v) / len(v), k) for k, v in ratios.items())[:5]
    print("Lowest actual/target thread ratios:",
          [(k[0], k[1], round(r, 2)) for r, k in worst])
    return verify(args.out)


if __name__ == "__main__":
    sys.exit(main())
