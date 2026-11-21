"""Does the algorithm choice survive other ways of counting?

Run from the repo root:  python paper/scripts/robustness.py

The main tables pool classes inside a cell (hits / classes), so a scene with many
classes weighs more, and they average over 2,000-20,000 threads, where cells hold
15 scenes up to 10,000 and 5 above. This script recomputes, for the VLM and
pretrained YOLO:
  pooled     the main definition
  per-scene  recall per scene, then the mean over the scenes of a cell
  balanced   only 2,000-10,000 threads, where every cell holds 15 scenes
  both       per-scene and balanced together
and reports the Spearman correlation of each variant's algorithm ranking with the
main one, the algorithm with the most recall per KB, and the carried thread count.
Writes paper/data/robustness.tex (macros) and prints a summary.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "recognition"))
from score import score_answer  # noqa: E402

D = ROOT / "paper" / "data"
VLM = ROOT / "paper" / "vlm"
ALGOS = (D / "order.txt").read_text().split()
ALL_T = list(range(2000, 20001, 2000))
BAL_T = list(range(2000, 10001, 2000))
labels = lambda p: [] if p == "NONE" else p.split("|")

kb = defaultdict(list)
for f in ("sizes.csv", "sizes_extra.csv"):
    if (D / f).exists():
        for r in csv.DictReader(open(D / f)):
            kb[(r["method"], int(r["threads"]))].append(int(r["xz_bytes"]) / 1024)
kb = {k: sum(v) / len(v) for k, v in kb.items()}

# per-scene (hits, classes) for each cell
scenes = {"vlm": defaultdict(list), "yolo": defaultdict(list)}
for gt_p, pred_p in ((ROOT / "IMAGE_DATASET/ground_truth.csv", VLM / "round1_thread_images.csv"),
                     (ROOT / "IMAGE_DATASET_EXTRA/ground_truth_extra.csv", VLM / "round2_thread_images.csv")):
    if not pred_p.exists():
        continue
    pred = {r["identifier"]: r["prediction"] for r in csv.DictReader(open(pred_p)) if r["prediction"]}
    for r in csv.DictReader(open(gt_p)):
        truth = {c for c in r["ground_truth"].split("|") if c}
        if truth and r["identifier"] in pred:
            a = score_answer(labels(pred[r["identifier"]]), truth)
            scenes["vlm"][(r["method"], int(r["threads"]))].append((a["hits"], a["truth"]))
for f in ("yolo_pre_round1.csv", "yolo_pre_extra.csv"):
    for r in csv.DictReader(open(D / f)):
        truth = {c for c in r["truth"].split("|") if c}
        found = {c for c in r["found_thread"].split("|") if c}
        if truth:
            scenes["yolo"][(r["algorithm"], int(r["threads"]))].append((len(truth & found), len(truth)))


def cell(model, m, t, per_scene):
    s = scenes[model][(m, t)]
    if not s:
        return 0.0
    if per_scene:
        return 100 * sum(h / n for h, n in s) / len(s)
    return 100 * sum(h for h, _ in s) / sum(n for _, n in s)


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: -x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


def variant(per_scene, threads):
    out = {}
    for model in ("vlm", "yolo"):
        rec = {m: sum(cell(model, m, t, per_scene) for t in threads) / len(threads) for m in ALGOS}
        per_kb = {m: sum(cell(model, m, t, per_scene) / kb[(m, t)] for t in threads) / len(threads) for m in ALGOS}
        out[model] = (rec, per_kb)
    best = max(ALGOS, key=lambda m: out["vlm"][1][m])
    best_y = max(ALGOS, key=lambda m: out["yolo"][1][m])
    tv = {t: cell("vlm", best, t, per_scene) / kb[(best, t)] for t in threads}
    ty = {t: cell("yolo", best, t, per_scene) / kb[(best, t)] for t in threads}
    mv, my = max(tv.values()) or 1, max(ty.values()) or 1
    carry = max(threads, key=lambda t: tv[t] / mv + ty[t] / my)
    return out, best, best_y, carry


main, main_best, main_best_y, main_carry = variant(False, ALL_T)
rows = []
for name, ps, th in (("pooled", False, ALL_T), ("per-scene", True, ALL_T),
                     ("balanced", False, BAL_T), ("both", True, BAL_T)):
    out, best, best_y, carry = variant(ps, th)
    rv = spearman([main["vlm"][0][m] for m in ALGOS], [out["vlm"][0][m] for m in ALGOS])
    ry = spearman([main["yolo"][0][m] for m in ALGOS], [out["yolo"][0][m] for m in ALGOS])
    rows.append((name, rv, ry, best, best_y, carry))
    print(f"{name:10s} rho VLM {rv:.2f}  rho YOLO {ry:.2f}  best per KB VLM {best:16s} YOLO {best_y:16s} carry {carry}")

min_rho = min(min(r[1], r[2]) for r in rows[1:])
same_best = all(r[3] == main_best and r[4] == main_best_y for r in rows)
carries = sorted({r[5] for r in rows})
(D / "robustness.tex").write_text(
    f"\\renewcommand{{\\RobustMinRho}}{{{min_rho:.2f}}}\n"
    f"\\renewcommand{{\\RobustSameBest}}{{{'yes' if same_best else 'no'}}}\n"
    f"\\renewcommand{{\\RobustCarries}}{{{', '.join(f'{c:,}' for c in carries)}}}\n")
print("min rho", round(min_rho, 2), "same best:", same_best, "carry counts:", carries)

# Thread counts the algorithms actually delivered, from the measured thread lists.
short = defaultdict(int); total = 0
for f in ("sizes.csv", "sizes_extra.csv"):
    if (D / f).exists():
        for r in csv.DictReader(open(D / f)):
            total += 1
            if int(r["n_threads"]) < 0.9 * int(r["threads"]):
                short[r["method"]] += 1
n_short = sum(short.values())
names = sorted(short, key=lambda m: -short[m])
with open(D / "robustness.tex", "a") as f:
    f.write(f"\\renewcommand{{\\ShortImages}}{{{n_short}}}\n"
            f"\\renewcommand{{\\ShortTotal}}{{{total:,}}}\n"
            f"\\renewcommand{{\\ShortAlgos}}{{{', '.join(m.replace('_', ' ') for m in names[:-1])}"
            f"{' and ' if len(names) > 1 else ''}{names[-1].replace('_', ' ') if names else 'none'}}}\n")
print("short of 90% target:", n_short, "of", total, dict(short))
