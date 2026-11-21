"""Balanced check, step 3: score pretrained YOLO on every scene x algorithm x count, 2,000 to 20,000 threads in steps of 2,000.

Run from the repo root with the project venv, after balanced_render.sh:
    .venv/bin/python paper/scripts/balanced_yolo.py <work_dir>

Every scene is drawn with every algorithm, so scene difficulty is the same for all
algorithms. Recall is the share of a scene's COCO classes that YOLOv12n finds at
confidence 0.3, per scene, then averaged. Drawing time is read from the threader logs.
Writes paper/data/balanced.csv, paper/data/balanced.tex (macros) and
paper/data/tab_balanced.tex (small table).
"""
import csv, re, sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "paper" / "data"
work = Path(sys.argv[1])
ALGOS = (D / "order.txt").read_text().split()
THREADS = list(range(2000, 20001, 2000))
scenes = {r["scene"]: {c for c in r["classes"].split("|") if c} for r in csv.DictReader(open(work / "scenes.csv"))}

model = YOLO(str(ROOT / "models" / "yolo12n.pt"))
names = model.names


def found(paths):
    out = []
    for k in range(0, len(paths), 64):
        for p in model.predict([str(x) for x in paths[k:k + 64]], conf=0.3, verbose=False):
            out.append({names[int(c)] for c in p.boxes.cls.tolist()})
    return out


def recall(truth, got):
    return 100 * len(truth & got) / len(truth)


# originals, for retention
orig = dict(zip(scenes, found([work / "src" / f"{s}.png" for s in scenes])))
orig_rec = {s: recall(scenes[s], orig[s]) for s in scenes}

# drawing time per algorithm and count, from the threader log lines "[k] name  N threads  X s"
pat = re.compile(r"\[\d+\]\s+(\w+)\s+(\d+) threads\s+([\d.]+) s")
secs = defaultdict(list)
for log in (work / "log").glob("*.txt"):
    t = int(log.name.split("_")[0])
    for m in pat.finditer(log.read_text()):
        secs[(m.group(1), t)].append(float(m.group(3)))

rows = []
for t in THREADS:
    items = [(s, a) for s in scenes for a in ALGOS]
    got = found([work / "r" / str(t) / s / f"{s}_{a}.png" for s, a in items])
    for (s, a), g in zip(items, got):
        rows.append({"scene": s, "algorithm": a, "threads": t, "recall": recall(scenes[s], g),
                     "orig_recall": orig_rec[s]})

with open(D / "balanced.csv", "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)

cell = defaultdict(list)
for r in rows:
    cell[(r["algorithm"], r["threads"])].append(r["recall"])
avg = {a: sum(sum(cell[(a, t)]) / len(cell[(a, t)]) for t in THREADS) / len(THREADS) for a in ALGOS}
base = sum(orig_rec.values()) / len(orig_rec)


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: -x[i]); r = [0.0] * len(x); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b); ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


# compare with the main (unbalanced) pretrained YOLO and VLM averages over the same counts
main = {"vlm": defaultdict(list), "yolo": defaultdict(list)}
for r in csv.DictReader(open(D / "efficiency_full.csv")):
    if int(r["threads"]) in THREADS:
        main["vlm"][r["algorithm"]].append(float(r["vlm_recall"]))
        main["yolo"][r["algorithm"]].append(float(r["yolo_recall"]))
rho_y = spearman([avg[a] for a in ALGOS], [sum(main["yolo"][a]) / len(THREADS) for a in ALGOS])
rho_v = spearman([avg[a] for a in ALGOS], [sum(main["vlm"][a]) / len(THREADS) for a in ALGOS])
order = sorted(ALGOS, key=lambda a: -avg[a])

with open(D / "tab_balanced.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Balanced check: every scene drawn by every algorithm, scored by YOLO.}\n\\label{tab:balanced}\n\\footnotesize\n"
            "\\begin{tabular}{lrr}\n\\toprule\nAlgorithm & Recall (\\%) & Time (s) \\\\\n\\midrule\n")
    for a in order:
        st = sorted(x for t in THREADS for x in secs[(a, t)])
        med = st[len(st) // 2] if st else float("nan")
        f.write(f"{a.replace('_', ' ')} & {avg[a]:.1f} & {med:.1f} \\\\\n")
    f.write(f"\\midrule\noriginal photos & {base:.1f} & -- \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")

(D / "balanced.tex").write_text(
    f"\\renewcommand{{\\BalScenes}}{{{len(scenes)}}}\n"
    f"\\renewcommand{{\\BalRhoYolo}}{{{rho_y:.2f}}}\n"
    f"\\renewcommand{{\\BalRhoVlm}}{{{rho_v:.2f}}}\n"
    f"\\renewcommand{{\\BalTop}}{{{order[0].replace('_', ' ')}}}\n"
    f"\\renewcommand{{\\BalBase}}{{{base:.0f}}}\n"
    f"\\renewcommand{{\\BalSecond}}{{{order[1].replace('_', ' ')}}}\n")
print("order:", order)
print(f"rho vs main pretrained YOLO {rho_y:.2f}, vs VLM {rho_v:.2f}, originals recall {base:.1f}")
