"""Rank the algorithms by recall per kilobyte, ignoring drawing time.

Reads paper/data/efficiency_full.csv (one row per algorithm and thread count)
and writes:
  paper/data/tab_size_only.tex       small table: recall per KB for both models,
                                     and the smallest file that reaches 30% VLM recall
  paper/data/heat_{vlm,yolo}_kb.dat  recall per KB for every algorithm and thread
                                     count, rows in the same order as the other heatmaps
  paper/data/size_only.tex           macros quoted in the text
"""

import csv
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
VLM_TARGET = 30.0

rows = list(csv.DictReader(open(DATA / "efficiency_full.csv")))
by_algo = defaultdict(list)
for r in rows:
    by_algo[r["algorithm"]].append(r)
threads = sorted({int(r["threads"]) for r in rows})


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def per_kb(r, key):
    return float(r[key]) / float(r["kb"])


def smallest(cells, key, target):
    for r in sorted(cells, key=lambda r: float(r["kb"])):
        if float(r[key]) >= target:
            return f"{float(r['kb']):.0f}"
    return "--"


table = []
for algo, cells in by_algo.items():
    table.append({
        "algo": algo,
        "name": algo.replace("_", " "),
        "vlm_kb": mean(per_kb(r, "vlm_recall") for r in cells),
        "yolo_kb": mean(per_kb(r, "yolo_recall") for r in cells),
        "vlm_min": smallest(cells, "vlm_recall", VLM_TARGET),
    })
table.sort(key=lambda t: t["vlm_kb"], reverse=True)
best_v = max(t["vlm_kb"] for t in table)
best_y = max(t["yolo_kb"] for t in table)


def bold(x, best):
    s = f"{x:.2f}"
    return f"\\textbf{{{s}}}" if abs(x - best) < 1e-9 else s


lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Recall per kilobyte, drawing time ignored. The last column is the smallest file reaching "
    + f"{VLM_TARGET:.0f}\\% VLM recall.}}",
    r"\label{tab:size-only}",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{lrrr}",
    r"\toprule",
    r" & \multicolumn{2}{c}{Recall per KB} & Smallest \\",
    r"\cmidrule(lr){2-3}",
    f"Algorithm & VLM & YOLO & KB, {VLM_TARGET:.0f}\\% VLM \\\\",
    r"\midrule",
]
for t in table:
    lines.append(f"{t['name']} & {bold(t['vlm_kb'], best_v)} & {bold(t['yolo_kb'], best_y)} & {t['vlm_min']} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(DATA / "tab_size_only.tex").write_text("\n".join(lines))

# Heatmaps of recall per KB. Rows follow the same order as heat_<model>_recall.dat
# (sorted by the model's average recall), so the maps line up across a row.
best_algo = {}
heat_y = {}
for name, key in (("vlm", "vlm_recall"), ("yolo", "yolo_recall")):
    order = sorted(by_algo, key=lambda a: -sum(float(r[key]) for r in by_algo[a]))
    best_algo[name] = max(by_algo, key=lambda a: mean(per_kb(r, key) for r in by_algo[a]))
    heat_y[name] = order.index(best_algo[name])
    with open(DATA / f"heat_{name}_kb.dat", "w") as f:
        f.write("x y z\n")
        for yi, a in enumerate(order):
            cell = {int(r["threads"]): r for r in by_algo[a]}
            for xi, t in enumerate(threads):
                f.write(f"{xi} {yi} {per_kb(cell[t], key):.3f}\n")
            f.write("\n")

# Thread count to carry forward with the best algorithm: the count whose recall
# per KB is highest for both models together, each model scaled by its own best.
best_cells = {int(r["threads"]): r for r in by_algo[table[0]["algo"]]}
top_v = max(per_kb(r, "vlm_recall") for r in best_cells.values())
top_y = max(per_kb(r, "yolo_recall") for r in best_cells.values())
carry_t = max(best_cells, key=lambda t: per_kb(best_cells[t], "vlm_recall") / top_v
              + per_kb(best_cells[t], "yolo_recall") / top_y)
carry = best_cells[carry_t]

(DATA / "size_only.tex").write_text(
    f"\\renewcommand{{\\CarryThreads}}{{{carry_t:,}}}\n"
    f"\\renewcommand{{\\CarryKB}}{{{float(carry['kb']):.0f}}}\n"
    f"\\renewcommand{{\\CarryVlm}}{{{float(carry['vlm_recall']):.0f}}}\n"
    f"\\renewcommand{{\\CarryYolo}}{{{float(carry['yolo_recall']):.0f}}}\n"
)
with open(DATA / "size_only.tex", "a") as f:
  f.write(
    f"\\renewcommand{{\\SizeOnlyBest}}{{{table[0]['name']}}}\n"
    f"\\renewcommand{{\\SizeOnlyBestVlm}}{{{best_v:.2f}}}\n"
    f"\\renewcommand{{\\SizeOnlyBestYolo}}{{{best_y:.2f}}}\n"
    f"\\renewcommand{{\\SizeOnlyVlmY}}{{{heat_y['vlm']}}}\n"
    f"\\renewcommand{{\\SizeOnlyYoloY}}{{{heat_y['yolo']}}}\n"
)
print("\n".join(lines))
print("best per KB:", best_algo, "rows:", heat_y)
