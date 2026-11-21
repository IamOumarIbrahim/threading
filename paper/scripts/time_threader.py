"""Time the threader on one fixed image, for every algorithm and thread count.

Run from the repo root:
    python paper/scripts/time_threader.py <threader_binary> <image> [repeats]

Use the timing build of the threader, which prints the C++ steady-clock time
of the threading step alone (ALGO_SECONDS) and of the shared edge-field setup
(SETUP_SECONDS). Loading the image and writing files are not counted. Renders
run one at a time, never in parallel, so they do not compete for the CPU. Each
combination is repeated (default 3) and the median is kept. The image is COCO val2017 000000579158 (640 x 480), which is not part
of the study set. Writes paper/data/gen_time.csv.
"""
import csv, platform, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
binary, image = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
ALGOS = ["anchor_web", "edge_tangent", "flow_streamlines", "greedy_error", "greedy_residual",
         "quadtree_hatch", "radon_peel", "scanline_dashes", "stipple_tour", "thread_painter"]
THREADS = list(range(2000, 20001, 2000))

out = ROOT / "paper" / "data" / "gen_time.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["algorithm", "threads", "seconds_median", "seconds_min", "seconds_max", "setup_seconds", "repeats"])
    for a in ALGOS:
        for t in THREADS:
            times, setup = [], []
            for _ in range(reps):
                with tempfile.TemporaryDirectory() as d:
                    p = subprocess.run([binary, image, "--threads", str(t), "--seed", "7", "--only", a, "--out", d],
                                       check=True, capture_output=True, text=True)
                    line = [l for l in p.stdout.replace("\r", "\n").split("\n") if "ALGO_SECONDS" in l][0]
                    times.append(float(line.split()[-1]))
                    setup.append(float([l for l in p.stdout.split("\n") if l.startswith("SETUP_SECONDS")][0].split()[1]))
            w.writerow([a, t, f"{statistics.median(times):.4f}", f"{min(times):.4f}", f"{max(times):.4f}", f"{statistics.median(setup):.4f}", reps])
            f.flush()
            print(a, t, f"{statistics.median(times):.2f}s", flush=True)
print("machine:", platform.processor() or platform.machine())
