"""Full cost-effectiveness table: every algorithm at every thread count.

Run from the repo root:  python paper/scripts/make_efficiency.py
For each algorithm and thread count (2,000 to 20,000):
  seconds   C++ time of the threading step on one fixed 640x480 image
            (paper/data/gen_time.csv, median of 3 runs)
  kb        mean size of the compressed thread list (paper/data/sizes*.csv)
  vlm       VLM recall on the thread images (%), all scenes answered so far
  yolo      pretrained YOLO recall on the thread images (%), confidence 0.3
  *_eff     recall / (kb x seconds): recall bought per kilobyte-second
Writes paper/data/efficiency_full.csv and paper/data/tab_efficiency.tex.
"""
import csv, os, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "recognition"))
from score import score_answer  # noqa: E402

D = ROOT / "paper" / "data"
VLM = ROOT / "paper" / "vlm"
ALGOS = (D / "order.txt").read_text().split()
THREADS = list(range(2000, 20001, 2000))
nice = lambda m: m.replace("_", " ")
labels = lambda p: [] if p == "NONE" else p.split("|")

sec = {(r["algorithm"], int(r["threads"])): float(r["seconds_median"]) for r in csv.DictReader(open(D / "gen_time.csv"))}

size = defaultdict(list)
for f in ("sizes.csv", "sizes_extra.csv"):
    if (D / f).exists():
        for r in csv.DictReader(open(D / f)):
            size[(r["method"], int(r["threads"]))].append(int(r["xz_bytes"]) / 1024)

vlm = defaultdict(lambda: [0, 0])
for gt_p, pred_p in ((ROOT / "IMAGE_DATASET/ground_truth.csv", VLM / "round1_thread_images.csv"),
                     (ROOT / "IMAGE_DATASET_EXTRA/ground_truth_extra.csv", VLM / "round2_thread_images.csv")):
    if not pred_p.exists():
        continue
    pred = {r["identifier"]: r["prediction"] for r in csv.DictReader(open(pred_p)) if r["prediction"]}
    for r in csv.DictReader(open(gt_p)):
        truth = {c for c in r["ground_truth"].split("|") if c}
        if truth and r["identifier"] in pred:
            a = score_answer(labels(pred[r["identifier"]]), truth)
            k = (r["method"], int(r["threads"])); vlm[k][0] += a["hits"]; vlm[k][1] += a["truth"]

yolo = defaultdict(lambda: [0, 0])
for f in ("yolo_pre_round1.csv", "yolo_pre_extra.csv"):
    for r in csv.DictReader(open(D / f)):
        truth = {c for c in r["truth"].split("|") if c}; found = {c for c in r["found_thread"].split("|") if c}
        k = (r["algorithm"], int(r["threads"])); yolo[k][0] += len(truth & found); yolo[k][1] += len(truth)

rows = []
for m in ALGOS:
    for t in THREADS:
        k = (m, t); kb = sum(size[k]) / len(size[k]); s = sec[k]
        v = 100 * vlm[k][0] / vlm[k][1] if vlm[k][1] else 0.0
        y = 100 * yolo[k][0] / yolo[k][1] if yolo[k][1] else 0.0
        rows.append(dict(m=m, t=t, s=s, kb=kb, v=v, y=y, ve=v / (kb * s), ye=y / (kb * s)))

with open(D / "efficiency_full.csv", "w") as f:
    f.write("algorithm,threads,seconds,kb,vlm_recall,yolo_recall,vlm_eff,yolo_eff\n")
    for r in rows:
        f.write(f"{r['m']},{r['t']},{r['s']:.4f},{r['kb']:.1f},{r['v']:.1f},{r['y']:.1f},{r['ve']:.2f},{r['ye']:.2f}\n")

# Pareto front per model over the ten algorithms, using each algorithm's
# average recall and average efficiency over all ten thread counts (the
# table's Average column). An algorithm is on the front if no other algorithm
# has both higher-or-equal average recall and higher-or-equal average
# efficiency (and is strictly better in one). The single best algorithm is the
# front point closest to the ideal corner (highest recall, highest efficiency),
# with recall scaled by its maximum and efficiency scaled as log10(1 + eff).
import math
avg = {}
for name, rk, ek in (("vlm", "v", "ve"), ("yolo", "y", "ye")):
    avg[name] = {m: (sum(r[rk] for r in rows if r["m"] == m) / len(THREADS),
                     sum(r[ek] for r in rows if r["m"] == m) / len(THREADS)) for m in ALGOS}
front = {}; best = {}
for name in ("vlm", "yolo"):
    A = avg[name]
    on = [m for m in ALGOS if not any(A[q][0] >= A[m][0] and A[q][1] >= A[m][1] and (A[q][0] > A[m][0] or A[q][1] > A[m][1]) for q in ALGOS)]
    rmax = max(v[0] for v in A.values()) or 1; emax = max(math.log10(1 + v[1]) for v in A.values()) or 1
    best[name] = min(on, key=lambda m: math.hypot(1 - A[m][0] / rmax, 1 - math.log10(1 + A[m][1]) / emax))
    front[name] = on
    with open(D / f"pareto_{name}.csv", "w") as f:
        f.write("label,recall,eff,front\n")
        for m in ALGOS:
            lab = nice(m) if A[m][0] >= 0.5 else ""      # zero-recall points would stack their labels
            f.write(f"{lab},{A[m][0]:.2f},{math.log10(1 + A[m][1]):.4f},{int(m in on)}\n")
    with open(D / f"pareto_{name}_front.csv", "w") as f:
        f.write("recall,eff\n")
        for m in sorted(on, key=lambda m: A[m][0]):
            f.write(f"{A[m][0]:.2f},{math.log10(1 + A[m][1]):.4f}\n")

with open(D / "tab_efficiency.tex", "w") as f:
    f.write("\\begin{table*}[t]\n\\centering\n\\caption{Recall and efficiency of every algorithm at every thread count. Each cell gives recall (\\%) / efficiency. "
            "Efficiency is recall divided by (file size $\\times$ drawing time), in \\% per KB-second, so higher means more recognition for less cost. "
            "File size is the compressed thread list, and drawing time is the C++ time of the threading step on one 640$\\times$480 image. "
            "The last column averages over all ten thread counts. In each column the highest recall is underlined and the highest efficiency is in bold. For each model, the circled average marks the algorithm with the best balance of recall and efficiency (Fig.~\\ref{fig:pareto}).}\n"
            "\\label{tab:efficiency}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.3pt}\n\\begin{tabular}{@{}l" + "r" * 10 + "|r@{}}\n\\toprule\n")
    f.write("Algorithm & " + " & ".join(f"{t // 1000}k" for t in THREADS) + " & Average \\\\\n")
    for title, rk, ek in (("VLM", "v", "ve"), ("YOLO (pretrained)", "y", "ye")):
        f.write(f"\\midrule\n\\multicolumn{{12}}{{@{{}}l}}{{\\textit{{{title}: recall / efficiency}}}} \\\\\n")
        for m in ALGOS:
            vals = []
            for t in THREADS:
                col = [r for r in rows if r["t"] == t]
                r = next(x for x in col if x["m"] == m)
                rec, eff = f"{r[rk]:.0f}", f"{r[ek]:.1f}"
                if r[rk] > 0 and rec == f"{max(x[rk] for x in col):.0f}":
                    rec = f"\\underline{{{rec}}}"
                if r[ek] > 0 and eff == f"{max(x[ek] for x in col):.1f}":
                    eff = f"\\textbf{{{eff}}}"
                vals.append(f"{rec}\\,/\\,{eff}")
            mine = [x for x in rows if x["m"] == m]
            avg_r = sum(x[rk] for x in mine) / len(mine); avg_e = sum(x[ek] for x in mine) / len(mine)
            all_r = {a: sum(x[rk] for x in rows if x["m"] == a) / len(THREADS) for a in ALGOS}
            all_e = {a: sum(x[ek] for x in rows if x["m"] == a) / len(THREADS) for a in ALGOS}
            ar, ae = f"{avg_r:.0f}", f"{avg_e:.1f}"
            if avg_r > 0 and ar == f"{max(all_r.values()):.0f}": ar = f"\\underline{{{ar}}}"
            if avg_e > 0 and ae == f"{max(all_e.values()):.1f}": ae = f"\\textbf{{{ae}}}"
            avg_cell = f"{ar}\\,/\\,{ae}"
            if best["vlm" if rk == "v" else "yolo"] == m:
                avg_cell = f"\\bestcell{{{avg_cell}}}"
            f.write(f"{nice(m)} & " + " & ".join(vals) + f" & {avg_cell} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
# Heatmaps that mirror the table: recall next to efficiency, for each model.
# Rows are sorted by the model's average recall and kept in the same order in
# both maps of a pair, so a reader can look straight across. Efficiency is shown
# on a log scale, log10(1 + efficiency), because it spans 0 to several hundred.
import math
heat_labels = {}
for name, rk, ek in (("vlm", "v", "ve"), ("yolo", "y", "ye")):
    order = sorted(ALGOS, key=lambda a: -sum(r[rk] for r in rows if r["m"] == a))
    heat_labels[name] = ",".join(nice(a) for a in order)
    for kind, key, tf in (("recall", rk, lambda x: x), ("eff", ek, lambda x: math.log10(1 + x))):
        with open(D / f"heat_{name}_{kind}.dat", "w") as f:
            f.write("x y z\n")
            for yi, a in enumerate(order):
                for xi, t in enumerate(THREADS):
                    f.write(f"{xi} {yi} {tf(next(r[key] for r in rows if r['m'] == a and r['t'] == t)):.3f}\n")
                f.write("\n")
by = {}
for name, rk in (("vlm", "v"), ("yolo", "y")):
    order = sorted(ALGOS, key=lambda a: -sum(r[rk] for r in rows if r["m"] == a))
    by[name] = order.index(best[name])
bv, byo = avg["vlm"][best["vlm"]], avg["yolo"][best["yolo"]]
(D / "pareto_best.tex").write_text(
    f"\\renewcommand{{\\VlmParetoY}}{{{by['vlm']}}}\\renewcommand{{\\YoloParetoY}}{{{by['yolo']}}}\n"
    f"\\renewcommand{{\\VlmParetoName}}{{{nice(best['vlm'])}}}\\renewcommand{{\\YoloParetoName}}{{{nice(best['yolo'])}}}\n"
    f"\\renewcommand{{\\VlmParetoRecall}}{{{bv[0]:.0f}}}\\renewcommand{{\\YoloParetoRecall}}{{{byo[0]:.0f}}}\n"
    f"\\renewcommand{{\\VlmParetoEff}}{{{bv[1]:.1f}}}\\renewcommand{{\\YoloParetoEff}}{{{byo[1]:.1f}}}\n"
    f"\\renewcommand{{\\VlmFrontN}}{{{len(front['vlm'])}}}\\renewcommand{{\\YoloFrontN}}{{{len(front['yolo'])}}}\n"
    f"\\renewcommand{{\\VlmParetoBestE}}{{{math.log10(1 + bv[1]):.4f}}}\\renewcommand{{\\VlmParetoBestR}}{{{bv[0]:.2f}}}\n"
    f"\\renewcommand{{\\YoloParetoBestE}}{{{math.log10(1 + byo[1]):.4f}}}\\renewcommand{{\\YoloParetoBestR}}{{{byo[0]:.2f}}}\n")
(D / "pareto_algo.txt").write_text(best["vlm"] + "\n")
for name in ("vlm", "yolo"):
    print(name.upper(), "best:", best[name], "| front:", front[name])
    for m in ALGOS: print(f"   {m:17s} recall {avg[name][m][0]:5.1f}  eff {avg[name][m][1]:6.1f}{'  <- front' if m in front[name] else ''}")
(D / "heat_labels.tex").write_text(
    f"\\renewcommand{{\\VlmHeatOrder}}{{{heat_labels['vlm']}}}\n\\renewcommand{{\\YoloHeatOrder}}{{{heat_labels['yolo']}}}\n")

print(len(rows), "rows")
