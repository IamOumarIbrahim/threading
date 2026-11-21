"""Build every data file, table and number the paper reads.

Run from the repo root:  python paper/scripts/make_data.py

One measure is used for both models throughout the paper:

    retention = score on thread images / score on the original photos (%)

  VLM  : score = recall against the COCO classes. Only scenes whose original
         photo has also been answered are used, so both scores cover the same
         scenes.
  YOLO : pretrained YOLOv12n, no fine-tuning. score = recall against the same
         COCO classes as the VLM, from the classes it finds at confidence 0.3
         (paper/data/yolo_pre_*.csv, written by yolo_pretrained.py).

Both models and the human raters see the same 1,000 thread images, at
2,000 to 20,000 threads in steps of 2,000, and every result uses that grid.

Inputs : IMAGE_DATASET/ground_truth.csv, IMAGE_DATASET/control_map.csv,
         reports/threaded_benchmark_results.md, paper/data/sizes.csv,
         VLM predictions for the thread images and the original photos
         (paper/vlm/round*_*.csv, or THREADED_PRED / ORIGINAL_PRED in the environment).
Outputs: paper/data/*.csv, *.dat, tab_*.tex, numbers.tex
"""
import csv, os, re, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "recognition"))
from score import score_answer, recognised  # noqa: E402

OUT = ROOT / "paper" / "data"
THREADED = Path(os.environ.get("THREADED_PRED", ROOT / "paper" / "vlm" / "round1_thread_images.csv"))
ORIGINAL = Path(os.environ.get("ORIGINAL_PRED", ROOT / "paper" / "vlm" / "round1_original_photos.csv"))
ALGOS = ["anchor_web", "edge_tangent", "flow_streamlines", "greedy_error", "greedy_residual",
         "quadtree_hatch", "radon_peel", "scanline_dashes", "stipple_tour", "thread_painter"]
VLM_T = list(range(2000, 20001, 2000))
COMMON = VLM_T          # one grid for every judge
nice = lambda m: m.replace("_", " ")
f1 = lambda x: f"{x:.1f}"

gt = {r["identifier"]: r for r in csv.DictReader(open(ROOT / "IMAGE_DATASET/ground_truth.csv"))}
truth = {i: {c for c in r["ground_truth"].split("|") if c} for i, r in gt.items()}
labels = lambda p: [] if p == "NONE" else p.split("|")

def load(p):
    if not p.exists():
        return {}
    return {r["identifier"]: r["prediction"] for r in csv.DictReader(open(p)) if r["prediction"]}

thr = load(THREADED)
cmap = {r["control_id"]: r["threaded_identifier"] for r in csv.DictReader(open(ROOT / "IMAGE_DATASET/control_map.csv"))}
org = {cmap[k]: v for k, v in load(ORIGINAL).items()}

# Second round: 500 extra scenes, 10 more per cell at 2,000-10,000 threads,
# answered in their own blind folders. Merged here so every result uses both rounds.
EXTRA = ROOT / "IMAGE_DATASET_EXTRA"
VLM = ROOT / "paper" / "vlm"
if (EXTRA / "ground_truth_extra.csv").exists() and (VLM / "round2_thread_images.csv").exists():
    for r in csv.DictReader(open(EXTRA / "ground_truth_extra.csv")):
        gt[r["identifier"]] = r
        truth[r["identifier"]] = {c for c in r["ground_truth"].split("|") if c}
    thr.update(load(VLM / "round2_thread_images.csv"))
    cmap_x = {r["control_id"]: r["threaded_identifier"] for r in csv.DictReader(open(EXTRA / "control_map_extra.csv"))}
    org.update({cmap_x[k]: v for k, v in load(VLM / "round2_original_photos.csv").items()})
paired_all = [i for i in gt if i in thr and i in org]

def ids(method=None, threads=None):
    out = [i for i in paired_all
           if (method is None or gt[i]["method"] == method)
           and (threads is None or int(gt[i]["threads"]) in (threads if isinstance(threads, list) else [threads]))]
    return out

def recall(sel, pred):
    a = [score_answer(labels(pred[i]), truth[i]) for i in sel]
    tru = sum(x["truth"] for x in a)
    return 100 * sum(x["hits"] for x in a) / tru if tru else 0.0

def vlm_ret(sel):
    o = recall(sel, org)
    return 100 * recall(sel, thr) / o if o else 0.0

# ---- YOLO retention: pretrained YOLOv12n, recall on thread images / recall on originals ----
ypre = {}
for f in ("yolo_pre_round1.csv", "yolo_pre_extra.csv"):
    for r in csv.DictReader(open(OUT / f)):
        ypre[r["image_id"]] = r
ythr = {i: {c for c in r["found_thread"].split("|") if c} for i, r in ypre.items()}
yorg = {i: {c for c in r["found_original"].split("|") if c} for i, r in ypre.items()}

def ysel(method=None, threads=None):
    ts = threads if isinstance(threads, list) else [threads]
    return [i for i in gt if i in ypre and truth[i]
            and (method is None or gt[i]["method"] == method)
            and (threads is None or int(gt[i]["threads"]) in ts)]

def yrecall(sel, found):
    tru = sum(len(truth[i]) for i in sel)
    return 100 * sum(len(truth[i] & found[i]) for i in sel) / tru if tru else 0.0

def yolo_ret_sel(sel):
    o = yrecall(sel, yorg)
    return 100 * yrecall(sel, ythr) / o if o else 0.0

yolo_ret = lambda m, ts: yolo_ret_sel(ysel(m, ts))
yolo_cell = lambda m, t: yolo_ret_sel(ysel(m, t))

# ---- file sizes ----
sizes = {r["image_id"]: r for r in csv.DictReader(open(OUT / "sizes.csv"))}
if (OUT / "sizes_extra.csv").exists():
    sizes.update({r["image_id"]: r for r in csv.DictReader(open(OUT / "sizes_extra.csv"))})
def kb(m, t, col="xz_bytes"):
    v = [int(sizes[i][col]) for i in gt if gt[i]["method"] == m and int(gt[i]["threads"]) == t]
    return statistics.mean(v) / 1024
jpeg_kb = statistics.mean(int(r["jpeg_bytes"]) for r in sizes.values() if "jpeg_bytes" in r) / 1024

# ---- 1. by algorithm, over the common thread counts; one fixed order for the paper ----
rows = []
for m in ALGOS:
    sel = ids(m, COMMON)
    rows.append(dict(m=m, n=len(sel), rt=recall(sel, thr), ro=recall(sel, org), vr=vlm_ret(sel),
                     yr=yolo_ret(m, COMMON), kb=statistics.mean(kb(m, t) for t in COMMON)))
rows.sort(key=lambda r: -r["vr"])
ORDER = [r["m"] for r in rows]
(OUT / "order.txt").write_text("\n".join(ORDER) + "\n")
DESC = {
    "thread_painter": "Adds the line that most reduces the colour error, then refines its end points",
    "greedy_residual": "Like thread painter, with candidate lines drawn from where the error is largest",
    "quadtree_hatch": "Splits the image into cells and hatches busy cells along their edges",
    "greedy_error": "Grows short rays from important points and keeps the most uniform one",
    "radon_peel": "Repeatedly removes the strongest straight line of ink from the image",
    "flow_streamlines": "Follows the edge direction field and lays long chords along it",
    "edge_tangent": "Places fixed short segments along strong edges",
    "scanline_dashes": "Places horizontal dashes row by row, one per unit of ink",
    "anchor_web": "Links nearby important points with short straight edges",
    "stipple_tour": "Scatters dots by darkness and joins them with one nearest-neighbour path",
}
with open(OUT / "tab_algorithms.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{The ten threading algorithms, ordered by VLM retention}\n\\label{tab:algorithms}\n\\footnotesize\n"
            "\\begin{tabular}{@{}lp{5.5cm}@{}}\n\\toprule\nAlgorithm & How it places threads \\\\\n\\midrule\n")
    for m in ORDER:
        f.write(f"{nice(m)} & {DESC[m]} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
with open(OUT / "by_algorithm.csv", "w") as f:
    f.write("label,vlm,yolo\n")
    for r in rows:
        f.write(f"{nice(r['m'])},{f1(r['vr'])},{f1(r['yr'])}\n")
with open(OUT / "tab_by_algorithm.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Retention by algorithm, averaged over all thread counts (2,000 to 20,000).}\n"
            "\\label{tab:by-algorithm}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
            " & \\multicolumn{3}{c}{VLM} & YOLO \\\\\n\\cmidrule(lr){2-4}\\cmidrule(l){5-5}\n"
            "Algorithm & Recall, & Recall, & Retention & Retention \\\\\n"
            " & thread & original & & \\\\\n"
            " & image (\\%) & photo (\\%) & (\\%) & (\\%) \\\\\n\\midrule\n")
    for r in rows:
        f.write(f"{nice(r['m'])} & {f1(r['rt'])} & {f1(r['ro'])} & {f1(r['vr'])} & {f1(r['yr'])} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

def ranks(v):
    o = sorted(range(len(v)), key=lambda i: -v[i]); r = [0] * len(v)
    for k, i in enumerate(o): r[i] = k + 1
    return r
ra, rb = ranks([r["vr"] for r in rows]), ranks([r["yr"] for r in rows])
rho = 1 - 6 * sum((a - b) ** 2 for a, b in zip(ra, rb)) / (10 * 99)

# ---- 2. retention by thread count, averaged over all algorithms ----
with open(OUT / "by_thread_count.csv", "w") as f:
    f.write("k,vlm\n")
    for t in VLM_T:
        f.write(f"{t // 1000},{f1(vlm_ret(ids(threads=t)))}\n")
with open(OUT / "yolo_by_thread_count.csv", "w") as f:
    f.write("k,yolo\n")
    for t in VLM_T:
        f.write(f"{t // 1000},{f1(yolo_ret_sel(ysel(threads=t)))}\n")

# ---- 3. heatmaps, each sorted by its own row mean (best at the top) ----
def heat(name, cols, val):
    rowmean = {m: statistics.mean(min(val(m, t), 100) for t in cols) for m in ALGOS}
    order = sorted(ALGOS, key=lambda m: -rowmean[m])
    with open(OUT / f"{name}_heatmap.dat", "w") as f:
        f.write("x y z\n")
        for yi, m in enumerate(order):
            for xi, t in enumerate(cols):
                f.write(f"{xi} {yi} {f1(min(val(m, t), 100))}\n")
            f.write("\n")
    (OUT / f"{name}_heatmap_labels.tex").write_text(",".join(nice(m) for m in order))
    return order
vlm_order = heat("vlm", VLM_T, lambda m, t: vlm_ret(ids(m, t)))
yolo_order = heat("yolo", VLM_T, yolo_cell)

# ---- 4. retention against file size (best, worst, and the mean of all algorithms) ----
def size_curve(name, ts, ret, order):
    best, worst = order[0], order[-1]
    with open(OUT / f"size_{name}.csv", "w") as f:
        f.write("kb_best,ret_best,kb_worst,ret_worst,kb_all,ret_all\n")
        for t in ts:
            f.write(f"{f1(kb(best, t))},{f1(min(ret(best, t), 100))},{f1(kb(worst, t))},{f1(min(ret(worst, t), 100))},"
                    f"{f1(statistics.mean(kb(m, t) for m in ALGOS))},{f1(statistics.mean(min(ret(m, t), 100) for m in ALGOS))}\n")
    return best, worst
# The VLM has five scenes per algorithm and thread count, so neighbouring thread
# counts are pooled in pairs (2+4k, 6+8k, ...) to give ten scenes per point.
PAIRS = [VLM_T[i:i + 2] for i in range(0, len(VLM_T), 2)]
_kb = kb
def kb_pair(m, p): return statistics.mean(_kb(m, t) for t in p)
def size_curve_pairs(name, ret, order):
    best, worst = order[0], order[-1]
    with open(OUT / f"size_{name}.csv", "w") as f:
        f.write("kb_best,ret_best,kb_worst,ret_worst,kb_all,ret_all\n")
        for p in PAIRS:
            f.write(f"{f1(kb_pair(best, p))},{f1(min(ret(best, p), 100))},{f1(kb_pair(worst, p))},{f1(min(ret(worst, p), 100))},"
                    f"{f1(statistics.mean(kb_pair(m, p) for m in ALGOS))},{f1(statistics.mean(min(ret(m, p), 100) for m in ALGOS))}\n")
    return best, worst
vb, vw = size_curve_pairs("vlm", lambda m, p: vlm_ret(ids(m, p)), vlm_order)
yb, yw = size_curve_pairs("yolo", lambda m, p: yolo_ret_sel(ysel(m, p)), yolo_order)

# ---- 5. VLM on original photos against thread images ----
no = {i: recognised(labels(org[i])) & truth[i] for i in paired_all}
nt = {i: recognised(labels(thr[i])) & truth[i] for i in paired_all}
named_o = sum(len(no[i]) for i in paired_all)
kept = sum(len(no[i] & nt[i]) for i in paired_all)
only_t = sum(len(nt[i] - no[i]) for i in paired_all)
named_t = sum(len(nt[i]) for i in paired_all)
with open(OUT / "control.csv", "w") as f:
    f.write("label,count\n")
    f.write(f"named on both,{kept}\nnamed only on the original photo,{named_o - kept}\nnamed only on the thread image,{only_t}\n")

# ---- 6. numbers quoted in the text ----
mean_ret_common = vlm_ret(ids(threads=COMMON))
with open(OUT / "numbers.tex", "w") as f:
    w = lambda k, v: f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    w("NumClassesPerImage", f1(statistics.mean(len(truth[i]) for i in gt)))
    w("YoloRecallOriginal", f"{yrecall(ysel(), yorg):.0f}")
    w("YoloRecallThread", f"{yrecall(ysel(), ythr):.0f}")
    w("YoloRetentionAll", f"{yolo_ret_sel(ysel()):.0f}")
    w("RankCorr", f"{rho:.2f}")
    w("JpegMeanKB", f"{jpeg_kb:.0f}")
    w("PngMeanKB", f"{statistics.mean(int(r['png_bytes']) for r in sizes.values() if 'png_bytes' in r) / 1024:.0f}")
    w("PairedScenes", len(paired_all))
    w("VlmRecallOriginal", f"{recall(paired_all, org):.0f}")
    w("VlmRecallThread", f"{recall(paired_all, thr):.0f}")
    w("VlmRetentionAll", f"{vlm_ret(paired_all):.0f}")
    w("NamedOriginal", named_o)
    w("KeptCount", kept)
    w("KeptPct", f"{100 * kept / named_o:.0f}")
    w("OnlyThread", only_t)
    w("OnlyThreadPct", f"{100 * only_t / named_t:.0f}" if named_t else "0")
    w("XzKBlow", f"{statistics.mean(kb(m, VLM_T[0]) for m in ALGOS):.0f}")
    w("XzKBhigh", f"{statistics.mean(kb(m, VLM_T[-1]) for m in ALGOS):.0f}")
    w("XzHighPctJpeg", f"{100 * statistics.mean(kb(m, VLM_T[-1]) for m in ALGOS) / jpeg_kb:.0f}")
    w("BestAlgo", nice(ORDER[0])); w("WorstAlgo", nice(ORDER[-1]))
    w("VlmBest", nice(vb)); w("VlmWorst", nice(vw)); w("YoloBest", nice(yb)); w("YoloWorst", nice(yw))
    same = sorted({r["method"] for r in sizes.values() if "png_match" in r} - {r["method"] for r in sizes.values() if r.get("png_match") == "0"})
    w("IdenticalMethods", len(same))
    w("VlmHeatLabels", ",".join(nice(m) for m in vlm_order))
    w("YoloHeatLabels", ",".join(nice(m) for m in yolo_order))
    w("LostCount", named_o - kept)
    # The paper's central comparison: what choosing the algorithm changes
    # against what adding threads changes. Thread gain: 2,000 to 10,000 threads
    # (five times more), where every cell holds 15 scenes.
    v2, v10 = vlm_ret(ids(threads=2000)), vlm_ret(ids(threads=10000))
    y2, y10 = yolo_ret_sel(ysel(threads=2000)), yolo_ret_sel(ysel(threads=10000))
    w("VlmGainThreads", f"{v10 - v2:.0f}"); w("YoloGainThreads", f"{y10 - y2:.0f}")
    v20, y20 = vlm_ret(ids(threads=20000)), yolo_ret_sel(ysel(threads=20000))
    w("VlmGainThreadsTen", f"{v20 - v2:.0f}"); w("YoloGainThreadsTen", f"{y20 - y2:.0f}")
    w("VlmGapAlgos", f"{rows[0]['vr'] - rows[-1]['vr']:.0f}")
    yr = sorted(r["yr"] for r in rows)
    w("YoloGapAlgos", f"{yr[-1] - yr[0]:.0f}")

print("paired", len(paired_all), "rho", round(rho, 2))
print("order", ORDER)
print("vlm heat", vlm_order); print("yolo heat", yolo_order)
for r in rows: print(f"{r['m']:18s} rt {r['rt']:5.1f} ro {r['ro']:5.1f} vlm_ret {r['vr']:5.1f} yolo_ret {r['yr']:5.1f} kb {r['kb']:5.1f}")
print(open(OUT / "numbers.tex").read())
