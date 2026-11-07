#!/usr/bin/env python3
"""
make_graphs.py

Reads the "Digital Filography Detection Benchmark Report" (a Markdown file)
and draws 12 different graphs from it. The images are saved as PNG files in
reports/figures/.

How to run:
    python analysis/make_graphs.py                   (finds the report in reports/ or results/)
    python analysis/make_graphs.py path/to/report.md (use a specific report file)

Needs: Python 3.9+, matplotlib, numpy, pandas
    pip install matplotlib numpy pandas

The script reads the numbers from the report, so you can run it again each
time the benchmark report is updated (for example when more greedy_residual
thread counts are added).
"""

from __future__ import annotations

import copy
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to image files, no window needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "figures"

REPORT_MARKER = "Raw Detection Performance"  # text used to recognise the report

METRICS = {"map50": "mAP50", "map5095": "mAP50-95", "prec": "Precision", "rec": "Recall"}
METRIC_COLORS = {"map50": "#1f4e79", "map5095": "#2e86c1", "prec": "#f39c12", "rec": "#27ae60"}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "p", "*"]

plt.rcParams.update(
    {
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "figure.facecolor": "white",
    }
)


# ----------------------------------------------------------------------------
# Step 1: find and read the report
# ----------------------------------------------------------------------------
def find_report() -> Path:
    """Return the report path from the command line, or search for it."""
    if len(sys.argv) > 1:
        given = Path(sys.argv[1]).expanduser()
        if given.is_file():
            return given
        sys.exit(f"Report file not found: {given}")

    candidates: list[Path] = []
    for folder in dict.fromkeys([ROOT / "reports", ROOT / "results", Path.cwd()]):
        for pattern in ("*.md", "*.markdown", "*.txt"):
            candidates.extend(folder.glob(pattern))

    matches = []
    for path in candidates:
        try:
            if REPORT_MARKER in path.read_text(encoding="utf-8", errors="ignore"):
                matches.append(path)
        except OSError:
            continue

    if not matches:
        sys.exit(
            "Could not find the benchmark report.\n"
            "Save the report (.md or .txt) in the same folder as make_graphs.py,\n"
            "or run: python make_graphs.py path/to/report.md"
        )
    return max(matches, key=lambda p: p.stat().st_mtime)  # newest one wins


NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def to_number(cell: str) -> float:
    """Take the first number in a cell: '1,000' -> 1000, '54.2 min' -> 54.2."""
    match = NUMBER_RE.search(cell)
    if not match:
        return float("nan")
    return float(match.group(0).replace(",", ""))


def clean_cell(cell: str) -> str:
    return re.sub(r"[*`]", "", cell).strip()


def read_markdown_tables(text: str) -> list[list[list[str]]]:
    """Return every Markdown table as a list of rows (each row is a list of cells)."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [clean_cell(c) for c in stripped.strip("|").split("|")]
            is_separator = all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells)
            if not is_separator:
                current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def header_to_key(header: str) -> str | None:
    """Map a table header to a short column name. Unknown headers return None."""
    h = header.lower().strip()
    if "base)" in h:  # the "(% Base)" columns are not needed
        return None
    if h.startswith("algorithm"):
        return "algo"
    if h == "threads" or "thread count" in h:
        return "threads"
    if "train time" in h:
        return "train_min"
    if "gen time" in h:
        return "gen_ms"
    if "gflops" in h:
        return "gflops"
    if h.startswith("val ") or h.startswith("test "):
        split = h.split()[0]
        if "50-95" in h:
            return f"{split}_map5095"
        if "map50" in h:
            return f"{split}_map50"
        if "prec" in h:
            return f"{split}_prec"
        if "recall" in h:
            return f"{split}_rec"
    return None


def table_to_frame(table: list[list[str]]) -> pd.DataFrame:
    keys = [header_to_key(h) for h in table[0]]
    records = []
    for row in table[1:]:
        if len(row) != len(keys):
            continue
        record = {}
        for key, cell in zip(keys, row):
            if key is None:
                continue
            record[key] = cell if key == "algo" else to_number(cell)
        records.append(record)
    return pd.DataFrame(records)


def load_report(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Return (results table, baseline row). Retention columns start with 'r_'."""
    text = path.read_text(encoding="utf-8", errors="ignore")

    raw_frames, cost_frames = [], []
    for table in read_markdown_tables(text):
        keys = [header_to_key(h) for h in table[0]]
        if "algo" not in keys or "threads" not in keys:
            continue
        if "train_min" in keys:
            raw_frames.append(table_to_frame(table))
        elif "gen_ms" in keys:
            cost_frames.append(table_to_frame(table))

    if not raw_frames:
        sys.exit("No detection table found in the report. Is this the right file?")

    raw = pd.concat(raw_frames, ignore_index=True)
    is_baseline = raw["algo"].str.contains("baseline", case=False, na=False)
    if not is_baseline.any():
        sys.exit("The report has no baseline row, so retention cannot be computed.")
    baseline = raw[is_baseline].iloc[0]
    raw = raw[~is_baseline].dropna(subset=["threads"]).copy()
    raw["threads"] = raw["threads"].astype(int)
    raw = raw.drop_duplicates(["algo", "threads"], keep="last")  # newest row wins

    value_cols = [f"{s}_{m}" for s in ("val", "test") for m in METRICS]
    missing = [c for c in value_cols if c not in raw.columns]
    if missing:
        sys.exit(f"Missing columns in the detection table: {missing}")

    cost_cols = ["gen_ms", "gflops"]
    if cost_frames:
        cost = pd.concat(cost_frames, ignore_index=True)
        cost = cost[~cost["algo"].str.contains("baseline", case=False, na=False)]
        cost = cost.dropna(subset=["threads"]).copy()
        cost["threads"] = cost["threads"].astype(int)
        cost = cost.drop_duplicates(["algo", "threads"], keep="last")
        keep = ["algo", "threads"] + [c for c in cost_cols if c in cost.columns]
        df = raw.merge(cost[keep], on=["algo", "threads"], how="left")
    else:
        df = raw.copy()
    for col in cost_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["fps"] = 1000.0 / df["gen_ms"]
    for col in value_cols:
        df[f"r_{col}"] = df[col] / baseline[col] * 100.0  # retention in %

    df = df.sort_values(["algo", "threads"]).reset_index(drop=True)
    return df, baseline


# ----------------------------------------------------------------------------
# Step 2: shared helpers
# ----------------------------------------------------------------------------
@dataclass
class Ctx:
    df: pd.DataFrame
    base: pd.Series
    order: list[str]  # algorithms ranked by test mAP50 at the common budget
    color: dict
    marker: dict
    ref: int  # largest thread budget that every algorithm has

    @property
    def has_cost(self) -> bool:
        return bool(self.df["gen_ms"].notna().any())

    def rows(self, algo: str) -> pd.DataFrame:
        return self.df[self.df["algo"] == algo]


def build_context(df: pd.DataFrame, base: pd.Series) -> Ctx:
    ref = int(df.groupby("algo")["threads"].max().min())
    at_ref = df[df["threads"] == ref].sort_values("test_map50", ascending=False)
    order = list(at_ref["algo"])
    order += [a for a in sorted(df["algo"].unique()) if a not in order]
    palette = plt.get_cmap("tab10")
    color = {a: palette(i % 10) for i, a in enumerate(order)}
    marker = {a: MARKERS[i % len(MARKERS)] for i, a in enumerate(order)}
    return Ctx(df, base, order, color, marker, ref)


def thousands(x, _pos=None) -> str:
    return f"{x / 1000:g}k"


def style_thread_axis(ax, ctx: Ctx) -> None:
    ax.set_xticks(sorted(ctx.df["threads"].unique()))
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.tick_params(axis="x", labelsize=9)


def algo_handles(ctx: Ctx) -> list[Line2D]:
    return [
        Line2D([0], [0], color=ctx.color[a], marker=ctx.marker[a], lw=2, ms=6, label=a)
        for a in ctx.order
    ]


def bottom_legend(fig, ctx: Ctx, extra_title: str | None = None) -> None:
    fig.legend(
        handles=algo_handles(ctx),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=5,
        title=extra_title,
        fontsize=10,
    )


def save(fig, filename: str) -> None:
    fig.savefig(OUT_DIR / filename)
    plt.close(fig)
    print(f"  saved {filename}")


def draw_heatmap(fig, ax, values, row_labels, col_labels, cmap_name, norm, fmt, cbar_label):
    """Annotated heatmap. NaN cells are drawn light grey."""
    cmap = copy.copy(plt.get_cmap(cmap_name))
    cmap.set_bad("#ececec")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            value = values[r, c]
            if np.isnan(value):
                continue
            red, green, blue, _ = cmap(norm(value))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            ax.text(
                c, r, fmt.format(value), ha="center", va="center", fontsize=8.5,
                color="white" if luminance < 0.5 else "black",
            )

    colorbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.04)
    colorbar.set_label(cbar_label)
    colorbar.outline.set_visible(False)


def first_crossing(x: np.ndarray, y: np.ndarray, target: float) -> float:
    """Smallest x where y first reaches the target (linear interpolation)."""
    for i in range(len(x)):
        if y[i] >= target:
            if i == 0:
                return float(x[0])
            x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
            return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
    return float("nan")


# ----------------------------------------------------------------------------
# Step 3: the graphs
# ----------------------------------------------------------------------------
def fig_retention_lines(ctx: Ctx) -> None:
    """Line chart: test mAP50 retention against thread budget."""
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    ax.axhline(100, color="black", ls="--", lw=1.2)
    ax.text(ctx.df["threads"].min(), 101.5, "Baseline: original images (100%)", fontsize=9)
    ax.axvline(ctx.ref, color="grey", ls=":", lw=1)
    ax.text(ctx.ref, 2, f"common budget ({thousands(ctx.ref)})", rotation=90,
            ha="right", va="bottom", fontsize=8, color="grey")

    for rank, algo in enumerate(ctx.order):
        d = ctx.rows(algo)
        strong = rank < 2  # the two best algorithms get thicker lines
        ax.plot(d["threads"], d["r_test_map50"], color=ctx.color[algo], marker=ctx.marker[algo],
                lw=2.8 if strong else 1.6, ms=6 if strong else 4.5,
                alpha=1.0 if strong else 0.85, label=algo)

    style_thread_axis(ax, ctx)
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Thread budget")
    ax.set_ylabel("Retention (% of baseline test mAP50)")
    ax.set_title("How much detection quality survives as the thread budget grows")
    ax.legend(title=f"Algorithm\n(ranked at {thousands(ctx.ref)})", loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    save(fig, "01_test_map50_retention_lines.png")


def fig_heatmap(ctx: Ctx) -> None:
    """Heatmap: algorithm by thread budget."""
    piv = ctx.df.pivot(index="algo", columns="threads", values="r_test_map50").reindex(ctx.order)
    values = piv.to_numpy(dtype=float)
    vmax = max(60.0, float(np.nanmax(values)))
    fig, ax = plt.subplots(figsize=(1.8 + 0.8 * values.shape[1], 1.8 + 0.5 * values.shape[0]),
                           constrained_layout=True)
    draw_heatmap(fig, ax, values, list(piv.index), [thousands(t) for t in piv.columns],
                 "viridis", Normalize(vmin=0, vmax=vmax), "{:.0f}",
                 "% of baseline test mAP50")
    ax.set_xlabel("Thread budget")
    ax.set_title("Test mAP50 retention by algorithm and thread budget (grey = not run)")
    save(fig, "02_heatmap_test_map50.png")


def fig_metric_panels(ctx: Ctx) -> None:
    """Small multiples: the four test metrics side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, (key, label) in zip(axes.flat, METRICS.items()):
        ax.axhline(100, color="black", ls="--", lw=1)
        for algo in ctx.order:
            d = ctx.rows(algo)
            ax.plot(d["threads"], d[f"r_test_{key}"], color=ctx.color[algo],
                    marker=ctx.marker[algo], lw=1.6, ms=4, alpha=0.9)
        ax.set_title(f"Test {label}")
        ax.set_ylim(0, 108)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        style_thread_axis(ax, ctx)
    for ax in axes[1]:
        ax.set_xlabel("Thread budget")
    for ax in axes[:, 0]:
        ax.set_ylabel("% of baseline")
    fig.suptitle("All four test metrics, as a share of the baseline", fontsize=16,
                 fontweight="bold")
    bottom_legend(fig, ctx)
    save(fig, "03_all_metrics_small_multiples.png")


def fig_metric_breakdown(ctx: Ctx) -> None:
    """Grouped bars: the four metrics for each algorithm at the common budget."""
    d = ctx.df[ctx.df["threads"] == ctx.ref].set_index("algo").reindex(ctx.order)
    fig, ax = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
    x = np.arange(len(ctx.order))
    width = 0.2
    for j, (key, label) in enumerate(METRICS.items()):
        bars = ax.bar(x + (j - 1.5) * width, d[f"r_test_{key}"].fillna(0).to_numpy(), width,
                      label=f"Test {label}", color=METRIC_COLORS[key])
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
    ax.axhline(100, color="black", ls="--", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(ctx.order, rotation=25, ha="right")
    ax.set_ylim(0, 125)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("% of baseline")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"What each algorithm keeps at {thousands(ctx.ref)} threads (test split)")
    ax.legend(ncol=4, loc="upper center")
    save(fig, f"04_metric_breakdown_at_{thousands(ctx.ref)}.png")


def fig_precision_recall(ctx: Ctx) -> None:
    """Connected scatter: how precision and recall move together as threads grow."""
    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)

    p = np.linspace(0.005, 1.0, 600)
    for f1 in (0.2, 0.4, 0.6, 0.8):  # iso-F1 guide curves
        with np.errstate(divide="ignore", invalid="ignore"):
            r = f1 * p / (2 * p - f1)
        ok = (2 * p > f1) & (r > 0) & (r <= 1)
        ax.plot(r[ok], p[ok], color="#c8c8c8", lw=0.9, ls="--", zorder=0)
        ax.text(f1 / (2 - f1), 1.015, f"F1 = {f1}", fontsize=8, color="#888888", ha="center")

    for algo in ctx.order:
        d = ctx.rows(algo)
        rec, prec = d["test_rec"].to_numpy(), d["test_prec"].to_numpy()
        ax.plot(rec, prec, color=ctx.color[algo], lw=1.4, alpha=0.7, zorder=2)
        ax.scatter(rec, prec, s=np.linspace(14, 90, len(d)), color=ctx.color[algo],
                   edgecolor="white", linewidth=0.6, zorder=3)
        if len(d) > 1:
            ax.annotate("", xy=(rec[-1], prec[-1]), xytext=(rec[-2], prec[-2]),
                        arrowprops=dict(arrowstyle="->", color=ctx.color[algo], lw=1.4),
                        zorder=3)

    ax.scatter(ctx.base["test_rec"], ctx.base["test_prec"], marker="*", s=380, color="black",
               zorder=5)
    ax.annotate("Baseline", (ctx.base["test_rec"], ctx.base["test_prec"]),
                xytext=(-12, -4), textcoords="offset points", ha="right", fontsize=10,
                fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("Test recall")
    ax.set_ylabel("Test precision")
    ax.set_title("Precision and recall paths (marker size grows with thread budget)")
    ax.legend(handles=algo_handles(ctx), loc="lower right", fontsize=9)
    save(fig, "05_precision_recall_paths.png")


def fig_cost_scaling(ctx: Ctx) -> None:
    """Line chart on log axes: how the cost grows with thread budget."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    panels = [
        (axes[0], "gen_ms", "Mean generation time (ms per image)", "Time to draw one image"),
        (axes[1], "gflops", "Computational complexity (GFLOPs)", "Compute per image"),
    ]
    for ax, col, ylabel, title in panels:
        for algo in ctx.order:
            d = ctx.rows(algo)
            ax.plot(d["threads"], d[col], color=ctx.color[algo], marker=ctx.marker[algo],
                    lw=1.8, ms=4.5, alpha=0.9)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.grid(which="minor", alpha=0.12)
        style_thread_axis(ax, ctx)
        ax.set_xlabel("Thread budget")
        ax.set_ylabel(ylabel + ", log scale")
        ax.set_title(title)
    fig.suptitle("The price of more threads", fontsize=16, fontweight="bold")
    bottom_legend(fig, ctx)
    save(fig, "06_cost_scaling.png")


def fig_pareto(ctx: Ctx) -> None:
    """Scatter with a Pareto frontier: quality against cost."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    panels = [
        (axes[0], "gen_ms", "Mean generation time (ms per image, log scale)",
         "Quality against drawing time"),
        (axes[1], "gflops", "Computational complexity (GFLOPs, log scale)",
         "Quality against compute"),
    ]
    max_threads = ctx.df["threads"].max()
    for ax, col, xlabel, title in panels:
        data = ctx.df.dropna(subset=[col, "r_test_map50"])
        for algo in ctx.order:
            d = data[data["algo"] == algo]
            ax.scatter(d[col], d["r_test_map50"], s=22 + 95 * d["threads"] / max_threads,
                       color=ctx.color[algo], marker=ctx.marker[algo], alpha=0.85,
                       edgecolor="white", linewidth=0.5, zorder=3)

        points = sorted(zip(data[col], data["r_test_map50"], data["algo"], data["threads"]),
                        key=lambda q: (q[0], -q[1]))
        front, best = [], -np.inf
        for x, y, algo, threads in points:  # keep points that beat everything cheaper
            if y > best:
                front.append((x, y, algo, threads))
                best = y
        fx = [q[0] for q in front] + [data[col].max() * 1.15]
        fy = [q[1] for q in front] + [front[-1][1]]
        ax.step(fx, fy, where="post", color="black", ls="--", lw=1.2, zorder=2)
        for i, (x, y, algo, threads) in enumerate(front):
            ax.annotate(f"{algo} {thousands(threads)}", (x, y),
                        xytext=(6, 6 if i % 2 == 0 else -12), textcoords="offset points",
                        fontsize=7.5)

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.margins(x=0.1)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Retention (% of baseline test mAP50)")
        ax.set_title(title)
    fig.suptitle("Is the extra cost worth it? (dashed line = best quality for a given cost; "
                 "marker size = thread budget)", fontsize=14, fontweight="bold")
    bottom_legend(fig, ctx)
    save(fig, "07_quality_vs_cost_pareto.png")


def fig_dumbbell(ctx: Ctx) -> None:
    """Dumbbell chart: first budget against the common budget, per algorithm."""
    t0 = int(ctx.df["threads"].min())
    start = ctx.df[ctx.df["threads"] == t0].set_index("algo")["r_test_map50"]
    end = ctx.df[ctx.df["threads"] == ctx.ref].set_index("algo")["r_test_map50"]
    algos = [a for a in end.sort_values().index if a in start.index and pd.notna(end[a])]

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    for y, algo in enumerate(algos):
        color = ctx.color[algo]
        ax.plot([start[algo], end[algo]], [y, y], color="#bbbbbb", lw=3, zorder=1)
        ax.scatter(start[algo], y, s=80, color="white", edgecolor=color, linewidth=2, zorder=3)
        ax.scatter(end[algo], y, s=95, color=color, zorder=3)

        later = ctx.df[(ctx.df["algo"] == algo) & (ctx.df["threads"] > ctx.ref)]
        if later.empty:
            ax.text(end[algo] + 1.2, y, f"+{end[algo] - start[algo]:.1f} pp", va="center",
                    fontsize=9)
        else:
            last = later.iloc[-1]
            ax.plot([end[algo], last["r_test_map50"]], [y, y], color=color, lw=3, alpha=0.35,
                    zorder=2)
            ax.scatter(last["r_test_map50"], y, s=95, marker="D", facecolor="white",
                       edgecolor=color, linewidth=2, zorder=3)
            ax.text(last["r_test_map50"] + 1.2, y,
                    f"+{last['r_test_map50'] - start[algo]:.1f} pp at {thousands(last['threads'])}",
                    va="center", fontsize=9)

    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, ctx.df["r_test_map50"].max() + 16)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Retention (% of baseline test mAP50)")
    ax.set_title(f"Gain from {thousands(t0)} to {thousands(ctx.ref)} threads (pp = percentage points)")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="black", markeredgewidth=2, ms=9, label=f"{thousands(t0)} threads"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="black", ms=9,
                   label=f"{thousands(ctx.ref)} threads"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
                   markeredgecolor="black", markeredgewidth=2, ms=8, label="largest budget run"),
        ],
        loc="lower right",
    )
    save(fig, "08_dumbbell_gain_first_to_common_budget.png")


def fig_val_vs_test(ctx: Ctx) -> None:
    """Scatter against the diagonal: do validation results hold on the test split?"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), constrained_layout=True)
    max_threads = ctx.df["threads"].max()
    for ax, key in zip(axes, ("map50", "map5095")):
        val = ctx.df[f"val_{key}"].to_numpy()
        test = ctx.df[f"test_{key}"].to_numpy()
        top = max(val.max(), test.max()) * 1.08
        ax.plot([0, top], [0, top], color="black", ls="--", lw=1)
        for algo in ctx.order:
            d = ctx.rows(algo)
            ax.scatter(d[f"val_{key}"], d[f"test_{key}"], s=20 + 70 * d["threads"] / max_threads,
                       color=ctx.color[algo], marker=ctx.marker[algo], alpha=0.85,
                       edgecolor="white", linewidth=0.5)
        gap = test - val
        corr = np.corrcoef(val, test)[0, 1]
        ax.text(0.04, 0.95,
                f"Pearson r = {corr:.3f}\nmean (test - val) = {gap.mean():+.4f}\n"
                f"mean |test - val| = {np.abs(gap).mean():.4f}",
                transform=ax.transAxes, va="top", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Validation {METRICS[key]}")
        ax.set_ylabel(f"Test {METRICS[key]}")
        ax.set_title(METRICS[key])
    fig.suptitle("Validation against test: points near the diagonal mean stable results",
                 fontsize=15, fontweight="bold")
    bottom_legend(fig, ctx)
    save(fig, "09_validation_vs_test.png")


def fig_threads_needed(ctx: Ctx) -> None:
    """Inverse view: threads needed to reach a chosen quality level."""
    top = float(np.floor(ctx.df["r_test_map50"].max()))
    targets = np.arange(10.0, top + 1.0, 1.0)
    fig, ax = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)
    ax.axhline(ctx.ref, color="grey", ls=":", lw=1)
    ax.text(targets[0], ctx.ref, f" common budget ({thousands(ctx.ref)})", va="bottom",
            fontsize=8, color="grey")

    for algo in ctx.order:
        d = ctx.rows(algo)
        x = d["threads"].to_numpy(float)
        y = d["r_test_map50"].to_numpy(float)
        needed = np.array([first_crossing(x, y, t) for t in targets])
        ax.plot(targets, needed, color=ctx.color[algo], lw=2, alpha=0.9)
        valid = np.where(~np.isnan(needed))[0]
        if len(valid):  # marker at the best quality that algorithm reached
            ax.scatter(targets[valid[-1]], needed[valid[-1]], color=ctx.color[algo],
                       marker=ctx.marker[algo], s=70, zorder=3, edgecolor="white")

    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Target quality (% of baseline test mAP50)")
    ax.set_ylabel("Threads needed")
    ax.set_title("Cheapest thread budget for a target quality (lines stop at each algorithm's best result)")
    ax.legend(handles=algo_handles(ctx), loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0)
    save(fig, "10_threads_needed_for_quality.png")


def fig_marginal_gain(ctx: Ctx) -> None:
    """Diverging heatmap: change per extra 1,000 threads (shows diminishing returns)."""
    piv = ctx.df.pivot(index="algo", columns="threads", values="r_test_map50").reindex(ctx.order)
    threads = piv.columns.to_numpy()
    if len(threads) < 2:
        print("  skipped marginal gain heatmap (needs at least two thread budgets)")
        return
    step = np.diff(threads) / 1000.0
    gain = piv.diff(axis=1).iloc[:, 1:].to_numpy(dtype=float) / step[None, :]
    norm = TwoSlopeNorm(vmin=min(float(np.nanmin(gain)), -0.5), vcenter=0.0,
                        vmax=max(float(np.nanmax(gain)), 0.5))
    fig, ax = plt.subplots(figsize=(1.8 + 0.8 * gain.shape[1], 1.8 + 0.5 * gain.shape[0]),
                           constrained_layout=True)
    draw_heatmap(fig, ax, gain, list(piv.index), [thousands(t) for t in threads[1:]],
                 "RdYlGn", norm, "{:+.1f}",
                 "Change in retention per +1k threads (percentage points)")
    ax.set_xlabel("Thread budget reached")
    ax.set_title("Marginal gain per extra 1,000 threads (red = quality dropped)")
    save(fig, "11_marginal_gain_heatmap.png")


def fig_bump(ctx: Ctx) -> None:
    """Bump chart: the ranking of the algorithms at each thread budget."""
    sets = [set(g["threads"]) for _, g in ctx.df.groupby("algo")]
    counts = sorted(set.intersection(*sets))
    if len(counts) < 2:
        print("  skipped bump chart (needs at least two shared thread budgets)")
        return
    piv = ctx.df.pivot(index="algo", columns="threads", values="test_map50")[counts]
    piv = piv.reindex(ctx.order)
    ranks = piv.rank(ascending=False, method="min")
    n_algos, n_cols = ranks.shape

    fig, ax = plt.subplots(figsize=(13, 7.5), constrained_layout=True)
    xs = np.arange(n_cols)
    for pos, algo in enumerate(ctx.order):
        r = ranks.loc[algo].to_numpy()
        color = ctx.color[algo]
        strong = pos < 2
        ax.plot(xs, r, color=color, lw=4 if strong else 2.5, alpha=1.0 if strong else 0.7,
                solid_capstyle="round", zorder=2)
        ax.scatter(xs, r, s=330, color=color, zorder=3)
        for x_pos, rank_value in zip(xs, r):
            ax.text(x_pos, rank_value, str(int(rank_value)), color="white", ha="center",
                    va="center", fontsize=8, fontweight="bold", zorder=4)
        ax.text(xs[0] - 0.3, r[0], algo, ha="right", va="center", fontsize=10, color=color)
        ax.text(xs[-1] + 0.3, r[-1], algo, ha="left", va="center", fontsize=10, color=color)

    ax.set_ylim(n_algos + 0.6, 0.4)  # rank 1 at the top
    ax.set_xlim(-2.4, n_cols - 1 + 2.4)
    ax.set_yticks([])
    ax.set_xticks(xs)
    ax.set_xticklabels([thousands(c) for c in counts])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Thread budget")
    ax.set_title("Ranking by test mAP50 at each thread budget (1 = best)")
    save(fig, "12_rank_bump_chart.png")


# ----------------------------------------------------------------------------
# Step 4: run everything
# ----------------------------------------------------------------------------
def main() -> None:
    report = find_report()
    print(f"Reading report: {report}")
    df, baseline = load_report(report)
    ctx = build_context(df, baseline)
    print(f"Found {df['algo'].nunique()} algorithms, {len(df)} runs, "
          f"common budget {ctx.ref:,} threads.")
    print(f"Saving images to: {OUT_DIR}")

    jobs = [
        ("line chart", fig_retention_lines, False),
        ("heatmap", fig_heatmap, False),
        ("small multiples", fig_metric_panels, False),
        ("grouped bars", fig_metric_breakdown, False),
        ("precision-recall paths", fig_precision_recall, False),
        ("cost scaling", fig_cost_scaling, True),
        ("Pareto scatter", fig_pareto, True),
        ("dumbbell", fig_dumbbell, False),
        ("validation vs test", fig_val_vs_test, False),
        ("threads needed", fig_threads_needed, False),
        ("marginal gain heatmap", fig_marginal_gain, False),
        ("bump chart", fig_bump, False),
    ]
    failed = 0
    for name, function, needs_cost in jobs:
        if needs_cost and not ctx.has_cost:
            print(f"  skipped {name} (no generation time table in the report)")
            continue
        try:
            function(ctx)
        except Exception:  # keep going, so one bad graph does not stop the rest
            failed += 1
            print(f"  FAILED {name}")
            traceback.print_exc()
    print("Done." if not failed else f"Done, but {failed} graph(s) failed.")


if __name__ == "__main__":
    main()
