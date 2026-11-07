#!/usr/bin/env python3
"""
Plot *normalized* mAP50 and F1-score (% of YOLOv12n baseline)
vs. thread count for Digital Filography Detection Benchmark.

Saves:
  - norm_map50_vs_threads.png
  - norm_f1_vs_threads.png
in reports/figures/.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# Data (Test set – absolute values)
# ------------------------------------------------------------------
threads = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])

# Absolute Test mAP50
map50_abs = {
    "greedy_residual":  [0.0640, 0.1151, 0.1339, 0.1601, 0.1744, 0.2084, 0.2558, 0.2753, 0.3321, 0.3952],
    "flow_streamlines": [0.0994, 0.1467, 0.1739, 0.2020, 0.2264, 0.2389, 0.2626, 0.2574, 0.2497, 0.2904],
    "scanline_dashes":  [0.0071, 0.0325, 0.0677, 0.0993, 0.1222, 0.1384, 0.1464, 0.1733, 0.1862, 0.2023],
    "radon_peel":       [0.0690, 0.1339, 0.1844, 0.1988, 0.2224, 0.2396, 0.2539, 0.2580, 0.2524, 0.2815],
    "stipple_tour":     [0.0391, 0.0624, 0.0791, 0.0926, 0.1219, 0.1355, 0.1642, 0.1552, 0.1815, 0.1752],
}

# Absolute Test Precision & Recall
prec_abs = {
    "greedy_residual":  [0.2493, 0.3498, 0.3964, 0.4507, 0.4499, 0.5025, 0.5357, 0.6244, 0.6529, 0.6376],
    "flow_streamlines": [0.3620, 0.4230, 0.4758, 0.4574, 0.5093, 0.5371, 0.5010, 0.5199, 0.5444, 0.5461],
    "scanline_dashes":  [0.0916, 0.1543, 0.2571, 0.3088, 0.3945, 0.3814, 0.3912, 0.4453, 0.4413, 0.4521],
    "radon_peel":       [0.3131, 0.4401, 0.4636, 0.4468, 0.5120, 0.4892, 0.5184, 0.5094, 0.5452, 0.5237],
    "stipple_tour":     [0.2019, 0.2611, 0.2580, 0.3174, 0.3679, 0.3819, 0.4441, 0.4039, 0.4790, 0.4166],
}

recall_abs = {
    "greedy_residual":  [0.0883, 0.1518, 0.1716, 0.1893, 0.2163, 0.2530, 0.2956, 0.3124, 0.3690, 0.4355],
    "flow_streamlines": [0.1227, 0.1820, 0.2086, 0.2409, 0.2693, 0.2796, 0.3057, 0.3037, 0.2988, 0.3373],
    "scanline_dashes":  [0.0124, 0.0397, 0.0897, 0.1335, 0.1530, 0.1726, 0.1829, 0.2138, 0.2282, 0.2356],
    "radon_peel":       [0.0855, 0.1690, 0.2203, 0.2411, 0.2568, 0.2829, 0.2961, 0.3054, 0.2927, 0.3200],
    "stipple_tour":     [0.0536, 0.0834, 0.1078, 0.1258, 0.1580, 0.1715, 0.2017, 0.1926, 0.2175, 0.2167],
}

# ------------------------------------------------------------------
# Baseline (YOLOv12n @ τ=0.3) – Test set
# ------------------------------------------------------------------
BASE_MAP50   = 0.8841
BASE_PREC    = 0.9167
BASE_RECALL  = 0.8934
BASE_F1      = 2 * BASE_PREC * BASE_RECALL / (BASE_PREC + BASE_RECALL)

# ------------------------------------------------------------------
# Normalized values (% of baseline)
# ------------------------------------------------------------------
map50_norm = {k: (np.array(v) / BASE_MAP50 * 100) for k, v in map50_abs.items()}

f1_norm = {}
for algo in prec_abs:
    p = np.array(prec_abs[algo])
    r = np.array(recall_abs[algo])
    f1 = 2 * p * r / (p + r + 1e-12)
    f1_norm[algo] = f1 / BASE_F1 * 100

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
algorithms = list(map50_norm.keys())
colors = {
    "greedy_residual":  "#1f77b4",
    "flow_streamlines": "#ff7f0e",
    "scanline_dashes":  "#2ca02c",
    "radon_peel":       "#d62728",
    "stipple_tour":     "#9467bd",
}
markers = {
    "greedy_residual":  "o",
    "flow_streamlines": "s",
    "scanline_dashes":  "^",
    "radon_peel":       "D",
    "stipple_tour":     "v",
}

script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "figures")

# ------------------------------------------------------------------
# Figure 1 – Normalized mAP50
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
for algo in algorithms:
    plt.plot(
        threads,
        map50_norm[algo],
        label=algo,
        color=colors[algo],
        marker=markers[algo],
        linewidth=2,
        markersize=7,
    )

plt.axhline(100, color="gray", linestyle="--", linewidth=1.5, label="YOLOv12n baseline (100%)")
plt.title("Normalized Test mAP50 vs. Number of Threads\n(% of YOLOv12n baseline)", fontsize=13)
plt.xlabel("Number of Threads", fontsize=12)
plt.ylabel("Normalized mAP50 (%)", fontsize=12)
plt.xticks(threads, rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Technique", fontsize=9)
plt.tight_layout()

map_path = os.path.join(script_dir, "norm_map50_vs_threads.png")
plt.savefig(map_path, dpi=300, bbox_inches="tight")
print(f"Saved: {map_path}")
plt.close()

# ------------------------------------------------------------------
# Figure 2 – Normalized F1-score
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
for algo in algorithms:
    plt.plot(
        threads,
        f1_norm[algo],
        label=algo,
        color=colors[algo],
        marker=markers[algo],
        linewidth=2,
        markersize=7,
    )

plt.axhline(100, color="gray", linestyle="--", linewidth=1.5, label="YOLOv12n baseline (100%)")
plt.title("Normalized Test F1-score vs. Number of Threads\n(% of YOLOv12n baseline)", fontsize=13)
plt.xlabel("Number of Threads", fontsize=12)
plt.ylabel("Normalized F1-score (%)", fontsize=12)
plt.xticks(threads, rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Technique", fontsize=9)
plt.tight_layout()

f1_path = os.path.join(script_dir, "norm_f1_vs_threads.png")
plt.savefig(f1_path, dpi=300, bbox_inches="tight")
print(f"Saved: {f1_path}")
plt.close()

print("Done.")