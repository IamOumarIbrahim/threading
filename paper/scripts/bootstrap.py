"""95% bootstrap confidence intervals for VLM retention.

Run from the repo root:  python paper/scripts/bootstrap.py
Scenes are resampled with replacement inside each group, keeping every thread
image paired with its own original photo, and retention is recomputed each time
(2,000 resamples, seed 1). Writes paper/data/ci_by_algorithm.csv and
paper/data/ci_by_thread_count.csv. Optional EXTRA_* paths add more scenes.
"""
import csv, os, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "recognition"))
from score import score_answer  # noqa: E402

OUT = ROOT / "paper" / "data"
REPS, SEED = 2000, 1
COMMON = [2000, 4000, 6000, 8000, 10000]


def load_scenes():
    """Return {scene_id: (method, threads, hits_thread, hits_original, n_truth)}."""
    labels = lambda p: [] if p == "NONE" else p.split("|")
    scenes = {}
    sets = [(ROOT / "IMAGE_DATASET/ground_truth.csv", ROOT / "IMAGE_DATASET/control_map.csv",
             os.environ.get("THREADED_PRED", str(ROOT / "paper/vlm/round1_thread_images.csv")),
             os.environ.get("ORIGINAL_PRED", str(ROOT / "paper/vlm/round1_original_photos.csv")))]
    ex = ROOT / "IMAGE_DATASET_EXTRA"
    if (ex / "ground_truth_extra.csv").exists():
        sets.append((ex / "ground_truth_extra.csv", ex / "control_map_extra.csv",
                     str(ROOT / "paper/vlm/round2_thread_images.csv"), str(ROOT / "paper/vlm/round2_original_photos.csv")))
    if os.environ.get("EXTRA_GT"):
        sets.append((Path(os.environ["EXTRA_GT"]), Path(os.environ["EXTRA_MAP"]),
                     os.environ["EXTRA_THREADED"], os.environ["EXTRA_ORIGINAL"]))
    for gt_p, map_p, thr_p, org_p in sets:
        gt = {r["identifier"]: r for r in csv.DictReader(open(gt_p))}
        thr = {r["identifier"]: r["prediction"] for r in csv.DictReader(open(thr_p)) if r["prediction"]}
        cmap = {r["control_id"]: r["threaded_identifier"] for r in csv.DictReader(open(map_p))}
        org = {cmap[k]: r for k, r in ((r["identifier"], r["prediction"]) for r in csv.DictReader(open(org_p))) if r}
        for i, r in gt.items():
            truth = {c for c in r["ground_truth"].split("|") if c}
            if not truth or i not in thr or i not in org:
                continue
            ht = score_answer(labels(thr[i]), truth)["hits"]
            ho = score_answer(labels(org[i]), truth)["hits"]
            scenes[i] = (r["method"], int(r["threads"]), ht, ho, len(truth))
    return scenes


def retention(rows):
    o = sum(r[3] for r in rows)
    return 100 * sum(r[2] for r in rows) / o if o else 0.0


def ci(rows, rng):
    est = retention(rows)
    boots = sorted(retention([rng.choice(rows) for _ in rows]) for _ in range(REPS))
    return est, boots[int(0.025 * REPS)], boots[int(0.975 * REPS) - 1]


scenes = load_scenes()
rng = random.Random(SEED)
order = (OUT / "order.txt").read_text().split()

with open(OUT / "ci_by_algorithm.csv", "w") as f:
    f.write("label,n,vlm,lo,hi,minus,plus\n")
    for m in order:
        rows = [v for v in scenes.values() if v[0] == m and v[1] in COMMON]
        e, lo, hi = ci(rows, rng)
        f.write(f"{m.replace('_', ' ')},{len(rows)},{e:.1f},{lo:.1f},{hi:.1f},{e - lo:.1f},{hi - e:.1f}\n")

with open(OUT / "ci_by_thread_count.csv", "w") as f:
    f.write("k,n,vlm,lo,hi,minus,plus\n")
    for t in range(2000, 20001, 2000):
        rows = [v for v in scenes.values() if v[1] == t]
        e, lo, hi = ci(rows, rng)
        f.write(f"{t // 1000},{len(rows)},{e:.1f},{lo:.1f},{hi:.1f},{e - lo:.1f},{hi - e:.1f}\n")

print(open(OUT / "ci_by_algorithm.csv").read())
print(open(OUT / "ci_by_thread_count.csv").read())
