#!/usr/bin/env python3
"""Score recognition answers against COCO classes.

Usage (from the project root):
    python recognition/score.py
    python recognition/score.py --config config/recognition.json --split test

Writes to <data_dir>/scores/:
    by_cell.csv    one row per algorithm x budget
    by_rater.csv   one row per rater
    agreement.csv  label overlap between raters of the same image

Scores are per image, on the set of COCO classes present. Answers are typed
freely, so each is matched to a COCO class by its words and common synonyms.
Answers that match no class stay in the raw data and count as neither hit nor
false alarm, so review them before reporting final numbers.
Images with no COCO objects are skipped for recall but kept for false alarms.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vocabulary import classes_in  # noqa: E402

DEFAULT_CONFIG = HERE.parent / "config" / "recognition.json"


def recognised(labels: list[str]) -> set[str]:
    return {c for label in labels for c in classes_in(label)}


def score_answer(labels: list[str], truth: set[str]) -> dict:
    said = recognised(labels)
    hits = len(said & truth)
    return {
        "hits": hits, "said": len(said), "truth": len(truth),
        "recognised": int(hits > 0),
        "false_alarm": int(not truth and bool(said)),
    }


def summarise(answers: list[dict]) -> dict:
    hits = sum(a["hits"] for a in answers)
    said = sum(a["said"] for a in answers)
    truth = sum(a["truth"] for a in answers)
    precision = hits / said if said else 0.0
    recall = hits / truth if truth else 0.0
    with_truth = [a for a in answers if a["truth"]]
    times = sorted(a["ms"] for a in answers if a["ms"] is not None)
    return {
        "answers": len(answers),
        "recognised_rate": round(sum(a["recognised"] for a in with_truth)
                                 / max(len(with_truth), 1), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4)
              if precision + recall else 0.0,
        "unidentified_rate": round(sum(a["unidentified"] for a in answers)
                              / max(len(answers), 1), 4),
        "median_ms": times[len(times) // 2] if times else "",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--split", choices=["train", "val", "test"],
                    help="score one benchmark split only")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    data_dir = Path(cfg["data_dir"])
    with open(data_dir / "manifest.csv", newline="", encoding="utf-8") as f:
        manifest = {r["image_id"]: r for r in csv.DictReader(f)}

    db = sqlite3.connect(data_dir / "responses.sqlite")
    db.row_factory = sqlite3.Row

    cells, raters = defaultdict(list), defaultdict(list)
    per_image = defaultdict(dict)
    for r in db.execute("SELECT * FROM responses"):
        item = manifest.get(r["image_id"])
        if item is None or (args.split and item["split"] != args.split):
            continue
        labels = json.loads(r["labels"])
        truth = set(filter(None, item["classes"].split("|")))
        answer = score_answer(labels, truth)
        answer.update(unidentified=r["unidentified"], ms=r["response_ms"])
        cells[(item["algorithm"], int(item["threads_target"]))].append(answer)
        raters[r["username"]].append(answer)
        per_image[r["image_id"]][r["username"]] = recognised(labels)

    out = data_dir / "scores"
    out.mkdir(exist_ok=True)
    write_csv(out / "by_cell.csv",
              [{"algorithm": a, "threads": t, **summarise(v)}
               for (a, t), v in sorted(cells.items())])
    write_csv(out / "by_rater.csv",
              [{"rater": u, **summarise(v)} for u, v in sorted(raters.items())])

    overlap = defaultdict(list)
    for answers in per_image.values():
        for u, v in combinations(sorted(answers), 2):
            union = answers[u] | answers[v]
            overlap[(u, v)].append(
                len(answers[u] & answers[v]) / len(union) if union else 1.0)
    write_csv(out / "agreement.csv",
              [{"rater_a": u, "rater_b": v, "images": len(s),
                "mean_jaccard": round(sum(s) / len(s), 4)}
               for (u, v), s in sorted(overlap.items())])

    print(f"Scored {sum(map(len, raters.values()))} answers -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
