#!/usr/bin/env python3
"""Master pipeline for Digital Filography Benchmark:
- 60% / 20% / 20% Train / Val / Test split on 5,000 images (seed 1)
- Pseudo-ground-truth generation with YOLOv12n at tau = 0.3
- Baseline evaluation on clean original images
- Sequential per-algorithm threading and training:
    10 algorithms x 10 thread counts (1000..10000) = 100 runs
    100 epochs, batch 16, seed 1
    Evaluation at tau = 0.3
- Resumable state and dynamic results appending
"""

import argparse
import os
import json
import time
import shutil
import random
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import yaml
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
THREADED_DATASET_DIR = RESULTS_DIR / "threaded_dataset"
RUNS_DIR = PROJECT_ROOT / "runs" / "threaded_experiments"
THREADER_EXE = PROJECT_ROOT / "build" / "threader.exe"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
THREADED_DATASET_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_FILE = RESULTS_DIR / "threaded_benchmark_progress.json"
RESULTS_CSV = RESULTS_DIR / "threaded_benchmark_results.csv"
RESULTS_MD = RESULTS_DIR / "threaded_benchmark_results.md"
SPLITS_FILE = RESULTS_DIR / "splits_seed1.json"
GT_LABELS_DIR = RESULTS_DIR / "yolo12n_labels"

ALGORITHMS = [
    ("greedy_residual", 1),
    ("flow_streamlines", 2),
    ("scanline_dashes", 3),
    ("radon_peel", 4),
    ("stipple_tour", 5),
    ("edge_tangent", 6),
    ("greedy_error", 7),
    ("quadtree_hatch", 8),
    ("anchor_web", 9),
    ("thread_painter", 10),
]

THREAD_COUNTS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
EXTENDED_THREAD_COUNTS = [11000, 12000, 13000, 14000, 15000]

# Optional daily pause window (--daily-pause): 3:00 PM (15:00) to 11:00 PM (23:00)
PAUSE_ENABLED = False
PAUSE_START_HOUR = 15
PAUSE_END_HOUR = 23

class PauseTrainingInterrupt(Exception):
    """Raised when training reaches the daily pause window at an epoch boundary."""
    pass

def is_in_pause_window() -> bool:
    """True if --daily-pause is on and local time is within the pause hours."""
    now = time.localtime()
    return PAUSE_ENABLED and PAUSE_START_HOUR <= now.tm_hour < PAUSE_END_HOUR

def wait_if_in_pause_window(context: str = "") -> float:
    """Sleeps and waits if current time is within the daily pause window (3 PM - 11 PM).
    Returns total seconds spent paused.
    """
    if not is_in_pause_window():
        return 0.0

    t_start = time.time()
    prefix = f"[{context}] " if context else ""
    log(f"{prefix}Entering daily pause window (3:00 PM - 11:00 PM). State saved. Process safely paused.")
    log(f"{prefix}Waiting until 11:00 PM (23:00) to resume...")

    while is_in_pause_window():
        time.sleep(30)

    t_paused = time.time() - t_start
    log(f"{prefix}Daily pause window ended (11:00 PM reached). Restoring state and resuming operations! (Paused for {t_paused / 3600:.2f} hours)")
    return t_paused

def create_pause_callback(algo_name: str, thread_count: int):
    """Ultralytics callback triggered at the end of each epoch after checkpoint has been saved."""
    def on_fit_epoch_end(trainer):
        if is_in_pause_window():
            epoch_num = trainer.epoch + 1
            log(f"[{algo_name} | {thread_count} threads] 3:00 PM pause window reached. State successfully saved to {trainer.last} at epoch {epoch_num}/{trainer.epochs}.")
            log(f"[{algo_name} | {thread_count} threads] Releasing GPU VRAM and pausing trainer until 11:00 PM...")
            for loader in (trainer.train_loader, trainer.test_loader):
                if hasattr(loader, "close"):
                    loader.close()
            raise PauseTrainingInterrupt("Daily pause window active (3 PM - 11 PM)")
    return on_fit_epoch_end

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def check_disk_space_gb():
    total, used, free = shutil.disk_usage(str(PROJECT_ROOT))
    return free / 1e9

# ============================================================================
# Step 1: Split 5,000 images (60% / 20% / 20%) with seed 1
# ============================================================================
def get_or_create_splits():
    if SPLITS_FILE.exists():
        log(f"Loading existing split manifest from {SPLITS_FILE}")
        with open(SPLITS_FILE, "r") as f:
            return json.load(f)

    log("Creating 60% / 20% / 20% train/val/test split on 5,000 images (seed 1)...")
    all_imgs = sorted([p.name for p in DATASET_DIR.glob("*.jpg")])
    if len(all_imgs) != 5000:
        log(f"Warning: expected 5000 images, found {len(all_imgs)}")

    rng = random.Random(1)
    shuffled = list(all_imgs)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * 0.60) # 3000
    n_val = int(n_total * 0.20)   # 1000
    n_total - n_train - n_val # 1000

    train_imgs = sorted(shuffled[:n_train])
    val_imgs = sorted(shuffled[n_train:n_train + n_val])
    test_imgs = sorted(shuffled[n_train + n_val:])

    # Strict assertion of no overlap
    set_train = set(train_imgs)
    set_val = set(val_imgs)
    set_test = set(test_imgs)
    assert len(set_train.intersection(set_val)) == 0, "Train and Val overlap!"
    assert len(set_train.intersection(set_test)) == 0, "Train and Test overlap!"
    assert len(set_val.intersection(set_test)) == 0, "Val and Test overlap!"

    splits = {
        "seed": 1,
        "n_total": n_total,
        "n_train": len(train_imgs),
        "n_val": len(val_imgs),
        "n_test": len(test_imgs),
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs,
    }

    with open(SPLITS_FILE, "w") as f:
        json.dump(splits, f, indent=2)

    log(f"Splits saved: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")
    return splits

# ============================================================================
# Step 2: Generate Ground Truth Labels using YOLOv12n at tau = 0.3
# ============================================================================
def generate_yolo12n_ground_truth(splits):
    GT_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    existing_labels = list(GT_LABELS_DIR.glob("*.txt"))
    if len(existing_labels) == len(splits["train"]) + len(splits["val"]) + len(splits["test"]):
        log(f"YOLOv12n ground truth labels already generated ({len(existing_labels)} files).")
        return

    log("Generating pseudo-ground-truth labels using YOLOv12n (tau = 0.3)...")
    model_path = MODELS_DIR / "yolo12n.pt"
    model = YOLO(str(model_path))

    all_images = splits["train"] + splits["val"] + splits["test"]
    batch_size = 32
    t0 = time.time()

    for i in range(0, len(all_images), batch_size):
        batch_fnames = all_images[i:i + batch_size]
        batch_paths = [str(DATASET_DIR / fn) for fn in batch_fnames]
        results = model(batch_paths, conf=0.3, device=0, verbose=False, batch=batch_size)

        for fname, res in zip(batch_fnames, results):
            stem = Path(fname).stem
            txt_path = GT_LABELS_DIR / f"{stem}.txt"
            lines = []
            for b in res.boxes:
                cls_id = int(b.cls[0].item())
                xywhn = b.xywhn[0].tolist()
                lines.append(f"{cls_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")
            txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if (i + batch_size) % 320 == 0 or (i + batch_size) >= len(all_images):
            log(f"Ground truth generation: {min(i + batch_size, len(all_images))}/{len(all_images)} images ({time.time() - t0:.1f}s)")

    log("Finished generating all YOLOv12n ground truth labels.")

def extract_metrics(res):
    if hasattr(res, "box") and res.box is not None:
        return {
            "map50": float(res.box.map50),
            "map50_95": float(res.box.map),
            "precision": float(res.box.mp),
            "recall": float(res.box.mr),
        }
    rd = getattr(res, "results_dict", {})
    return {
        "map50": float(rd.get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(rd.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(rd.get("metrics/precision(B)", 0.0)),
        "recall": float(rd.get("metrics/recall(B)", 0.0)),
    }

# ============================================================================
# Step 3: Evaluate Baseline (Original Unthreaded Images) at tau = 0.3
# ============================================================================
def yolo_class_names() -> dict:
    """Class names in the id order used by the pseudo-ground-truth labels."""
    return dict(YOLO(str(MODELS_DIR / "yolo12n.pt")).names)

def evaluate_baseline(splits):
    baseline_file = RESULTS_DIR / "baseline_yolo12n_tau0.3.json"
    if baseline_file.exists():
        log(f"Baseline metrics already computed: {baseline_file}")
        with open(baseline_file, "r") as f:
            return json.load(f)

    log("Evaluating baseline performance of YOLOv12n on original unthreaded dataset (tau = 0.3)...")
    baseline_dir = RESULTS_DIR / "baseline_dataset"
    (baseline_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (baseline_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
    (baseline_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (baseline_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)

    # Link images and copy GT labels
    for fn in splits["val"]:
        src_img = DATASET_DIR / fn
        dst_img = baseline_dir / "images" / "val" / fn
        if not dst_img.exists():
            try:
                os.link(src_img, dst_img)
            except Exception:
                shutil.copy(src_img, dst_img)
        stem = Path(fn).stem
        src_lbl = GT_LABELS_DIR / f"{stem}.txt"
        dst_lbl = baseline_dir / "labels" / "val" / f"{stem}.txt"
        if not dst_lbl.exists():
            shutil.copy(src_lbl, dst_lbl)

    for fn in splits["test"]:
        src_img = DATASET_DIR / fn
        dst_img = baseline_dir / "images" / "test" / fn
        if not dst_img.exists():
            try:
                os.link(src_img, dst_img)
            except Exception:
                shutil.copy(src_img, dst_img)
        stem = Path(fn).stem
        src_lbl = GT_LABELS_DIR / f"{stem}.txt"
        dst_lbl = baseline_dir / "labels" / "test" / f"{stem}.txt"
        if not dst_lbl.exists():
            shutil.copy(src_lbl, dst_lbl)

    class_names = yolo_class_names()

    data_yaml = baseline_dir / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump({
            "path": str(baseline_dir.resolve()).replace("\\", "/"),
            "train": "images/val",
            "val": "images/val",
            "test": "images/test",
            "names": class_names
        }, f)

    model = YOLO(str(MODELS_DIR / "yolo12n.pt"))
    val_res = model.val(data=str(data_yaml), split="val", conf=0.3, batch=32, device=0, verbose=False)
    test_res = model.val(data=str(data_yaml), split="test", conf=0.3, batch=32, device=0, verbose=False)

    baseline_data = {
        "model": "yolo12n.pt",
        "tau": 0.3,
        "val": extract_metrics(val_res),
        "test": extract_metrics(test_res)
    }

    with open(baseline_file, "w") as f:
        json.dump(baseline_data, f, indent=2)

    log(f"Baseline computed: Val mAP50={baseline_data['val']['map50']:.4f}, Test mAP50={baseline_data['test']['map50']:.4f}")
    return baseline_data

# ============================================================================
# Step 4: Theoretical GFLOPs and Per-Image Generation Benchmarking
# ============================================================================
def calculate_gflops(algo_id: int, thread_count: int, width: int = 640, height: int = 480) -> float:
    """Theoretical floating-point operations (in GFLOPs) per image for given algo & thread count."""
    pixels = width * height
    n = thread_count
    if algo_id == 1:    # greedy_residual: 60 cands * 30 samples * 35 FLOPs + 3000 rasterization
        flops = 15 * pixels + n * (60 * 30 * 35 + 3000)
    elif algo_id == 2:  # flow_streamlines: structure tensor 150 FLOPs/pix + streamline Euler integration
        flops = 150 * pixels + n * 800
    elif algo_id == 3:  # scanline_dashes: 1D cumsum + dash placement
        flops = 4 * pixels + n * 50
    elif algo_id == 4:  # radon_peel: Radon transform 120 angles + iterative chord extraction
        diag = (width**2 + height**2)**0.5
        flops = 120 * diag * width + n * (120 * diag * 0.1)
    elif algo_id == 5:  # stipple_tour: blur 30 FLOPs/pix + nearest neighbor tour
        flops = 30 * pixels + n * 14 * 20
    elif algo_id == 6:  # edge_tangent: gradient field 35 FLOPs/pix + 250 FLOPs/thread
        flops = 35 * pixels + n * 250
    elif algo_id == 7:  # greedy_error: importance map 10 FLOPs/pix + 8 rays * 3 samples * 12 FLOPs
        flops = 10 * pixels + n * 388
    elif algo_id == 8:  # quadtree_hatch: variance tree 20 FLOPs/pix + hatch drawing
        flops = 20 * pixels + n * 40
    elif algo_id == 9:  # anchor_web: spatial sort + distance pruning + opacity ranking
        flops = 15 * pixels + (0.45 * n) * 12 * 10 + (1.35 * n) * 150
    elif algo_id == 10: # thread_painter: 160 cands * 48 samples * 35 + 144 mutants * 48 samples * 35 + antialiased kernel
        flops = 25 * pixels + n * (160 * 48 * 35 + 144 * 48 * 35 + 5000)
    else:
        flops = n * 1000
    return flops / 1e9

def benchmark_generation_time(algo_id: int, algo_name: str, thread_count: int, sample_size: int = 5) -> float:
    """Benchmark mean generation time in milliseconds per image using C++ threader.exe."""
    sample_imgs = sorted([p.name for p in DATASET_DIR.glob("*.jpg")])[:sample_size]
    times = []
    tmp_out = RESULTS_DIR / "scratch_bench"
    tmp_out.mkdir(parents=True, exist_ok=True)
    for fn in sample_imgs:
        cmd = [
            str(THREADER_EXE),
            str(DATASET_DIR / fn),
            "--only", str(algo_id),
            "--threads", str(thread_count),
            "--out", str(tmp_out)
        ]
        t0 = time.perf_counter()
        subprocess.run(cmd, capture_output=True, text=True)
        times.append(time.perf_counter() - t0)
    shutil.rmtree(tmp_out, ignore_errors=True)
    return (sum(times) / len(times)) * 1000.0 if times else 50.0

# ============================================================================
# Step 5: Parallel Threading for a Given Algorithm & Thread Count
# ============================================================================
def render_dataset_split(algo_id: int, algo_name: str, thread_count: int, split_name: str, file_names: list, target_dir: Path) -> list:
    wait_if_in_pause_window(f"{algo_name} | {thread_count} threads")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_name = lambda fn: f"{Path(fn).stem}.png"  # threader writes PNG
    missing = [fn for fn in file_names if not (target_dir / target_name(fn)).exists()]
    durations = []
    if not missing:
        return durations

    log(f"[{algo_name} | {thread_count} threads] Rendering {len(missing)} images for {split_name} split via C++ threader.exe...")

    def render_one(fn):
        wait_if_in_pause_window(f"{algo_name} | {thread_count} threads")
        src_path = str(DATASET_DIR / fn)
        out_folder = str(target_dir)
        cmd = [
            str(THREADER_EXE),
            src_path,
            "--only", str(algo_id),
            "--threads", str(thread_count),
            "--out", out_folder
        ]
        t0 = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True)
        t_dur = time.perf_counter() - t0
        if res.returncode != 0:
            raise RuntimeError(f"threader failed on {fn}: {res.stderr[-300:]}")

        stem = Path(fn).stem
        prod_path = target_dir / f"{stem}_{algo_name}.png"
        final_path = target_dir / target_name(fn)
        if prod_path.exists():
            if final_path.exists() and prod_path != final_path:
                final_path.unlink()
            shutil.move(prod_path, final_path)
        svg_path = target_dir / f"{stem}_{algo_name}.svg"
        if svg_path.exists():
            svg_path.unlink()
        return t_dur

    with ThreadPoolExecutor(max_workers=8) as ex:
        durations = list(ex.map(render_one, missing))
    return durations

# ============================================================================
# Step 6: Dynamic Results Reporting & Finalized Tables
# ============================================================================
def append_result(algo_name: str, algo_id: int, thread_count: int, val_metrics: dict, test_metrics: dict,
                  elapsed_min: float, mean_gen_ms: float, gflops: float):
    progress = {}
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r") as f:
                progress = json.load(f)
        except Exception:
            pass

    key = f"{algo_name}_{thread_count}"
    progress[key] = {
        "algo": algo_name,
        "algo_id": algo_id,
        "threads": thread_count,
        "val": val_metrics,
        "test": test_metrics,
        "elapsed_min": elapsed_min,
        "mean_gen_ms": mean_gen_ms,
        "gflops": gflops,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

    # Write CSV
    csv_header = "algo,algo_id,threads,val_mAP50,val_mAP50_95,val_precision,val_recall,test_mAP50,test_mAP50_95,test_precision,test_recall,mean_gen_ms,gflops,train_elapsed_min,timestamp\n"
    lines = [csv_header]
    for k, v in progress.items():
        vm = v["val"]
        tm = v["test"]
        lines.append(
            f"{v['algo']},{v.get('algo_id', 0)},{v['threads']},"
            f"{vm.get('map50', 0):.4f},{vm.get('map50_95', 0):.4f},{vm.get('precision', 0):.4f},{vm.get('recall', 0):.4f},"
            f"{tm.get('map50', 0):.4f},{tm.get('map50_95', 0):.4f},{tm.get('precision', 0):.4f},{tm.get('recall', 0):.4f},"
            f"{v.get('mean_gen_ms', 0):.1f},{v.get('gflops', 0):.4f},"
            f"{v.get('elapsed_min', 0):.1f},{v.get('timestamp', '')}\n"
        )
    RESULTS_CSV.write_text("".join(lines), encoding="utf-8")

    # Generate Markdown report with Finalized Tables & Normalized Performance
    baseline_info = ""
    bval = {}
    btest = {}
    baseline_file = RESULTS_DIR / "baseline_yolo12n_tau0.3.json"
    if baseline_file.exists():
        try:
            with open(baseline_file, "r") as f:
                bdata = json.load(f)
            bval = bdata.get("val", {})
            btest = bdata.get("test", {})
            baseline_info = (
                f"\n### Baseline (Original Unthreaded Images @ $\\tau = 0.3$)\n"
                f"- **Val mAP50**: `{bval.get('map50', 0):.4f}` | **Val mAP50-95**: `{bval.get('map50_95', 0):.4f}` | **Prec**: `{bval.get('precision', 0):.4f}` | **Recall**: `{bval.get('recall', 0):.4f}`\n"
                f"- **Test mAP50**: `{btest.get('map50', 0):.4f}` | **Test mAP50-95**: `{btest.get('map50_95', 0):.4f}` | **Prec**: `{btest.get('precision', 0):.4f}` | **Recall**: `{btest.get('recall', 0):.4f}`\n"
            )
        except Exception:
            pass

    md_lines = [
        "# Digital Filography Detection Benchmark Report",
        "",
        "> **Model**: `yolo12n.pt` (Pseudo-Ground-Truth: YOLOv12n baseline detections at $\\tau = 0.3$)",
        "> **Dataset**: 5,000 images (60% Train [3,000], 20% Val [1,000], 20% Test [1,000], Seed 1)",
        f"> **Last Updated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        baseline_info,
        "## Raw Detection Performance Across Algorithms and Thread Budgets",
        "",
        "| Algorithm | Threads | Val mAP50 | Val mAP50-95 | Val Prec | Val Recall | Test mAP50 | Test mAP50-95 | Test Prec | Test Recall | Train Time |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    # Baseline Row in Raw Table
    if bval and btest:
        md_lines.append(
            f"| **`Baseline (Original YOLOv12n)`** | **—** | "
            f"**{bval.get('map50', 0):.4f}** | **{bval.get('map50_95', 0):.4f}** | **{bval.get('precision', 0):.4f}** | **{bval.get('recall', 0):.4f}** | "
            f"**{btest.get('map50', 0):.4f}** | **{btest.get('map50_95', 0):.4f}** | **{btest.get('precision', 0):.4f}** | **{btest.get('recall', 0):.4f}** | "
            f"**—** |"
        )

    for k, v in progress.items():
        vm = v["val"]
        tm = v["test"]
        md_lines.append(
            f"| **`{v['algo']}`** | {v['threads']:,} | "
            f"{vm.get('map50', 0):.4f} | {vm.get('map50_95', 0):.4f} | {vm.get('precision', 0):.4f} | {vm.get('recall', 0):.4f} | "
            f"{tm.get('map50', 0):.4f} | {tm.get('map50_95', 0):.4f} | {tm.get('precision', 0):.4f} | {tm.get('recall', 0):.4f} | "
            f"{v.get('elapsed_min', 0):.1f} min |"
        )

    # Table 2: Normalized Table relative to YOLOv12n baseline
    md_lines.extend([
        "",
        "---",
        "",
        "## Normalized Detection Performance (% of YOLOv12n Baseline @ $\\tau = 0.3$)",
        "",
        "> Metric Retention: $\\text{Retention (\\%)} = \\frac{\\text{Metric}_{\\text{threaded}}}{\\text{Metric}_{\\text{baseline}}} \\times 100\\%$",
        "",
        "| Algorithm | Threads | Norm Val mAP50 | Norm Val mAP50-95 | Norm Val Prec | Norm Val Recall | Norm Test mAP50 | Norm Test mAP50-95 | Norm Test Prec | Norm Test Recall |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    if bval and btest:
        md_lines.append(
            "| **`Baseline (Original YOLOv12n)`** | **—** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |"
        )

    for k, v in progress.items():
        vm = v["val"]
        tm = v["test"]
        bv_m50 = bval.get("map50", 1.0) or 1.0
        bv_m95 = bval.get("map50_95", 1.0) or 1.0
        bv_p   = bval.get("precision", 1.0) or 1.0
        bv_r   = bval.get("recall", 1.0) or 1.0

        bt_m50 = btest.get("map50", 1.0) or 1.0
        bt_m95 = btest.get("map50_95", 1.0) or 1.0
        bt_p   = btest.get("precision", 1.0) or 1.0
        bt_r   = btest.get("recall", 1.0) or 1.0

        norm_v_m50 = (vm.get("map50", 0.0) / bv_m50) * 100.0
        norm_v_m95 = (vm.get("map50_95", 0.0) / bv_m95) * 100.0
        norm_v_p   = (vm.get("precision", 0.0) / bv_p) * 100.0
        norm_v_r   = (vm.get("recall", 0.0) / bv_r) * 100.0

        norm_t_m50 = (tm.get("map50", 0.0) / bt_m50) * 100.0
        norm_t_m95 = (tm.get("map50_95", 0.0) / bt_m95) * 100.0
        norm_t_p   = (tm.get("precision", 0.0) / bt_p) * 100.0
        norm_t_r   = (tm.get("recall", 0.0) / bt_r) * 100.0

        md_lines.append(
            f"| **`{v['algo']}`** | {v['threads']:,} | "
            f"{norm_v_m50:.1f}% | {norm_v_m95:.1f}% | {norm_v_p:.1f}% | {norm_v_r:.1f}% | "
            f"{norm_t_m50:.1f}% | {norm_t_m95:.1f}% | {norm_t_p:.1f}% | {norm_t_r:.1f}% |"
        )

    # Table 3: Finalized Table for Computational Efficiency & GFLOPs
    md_lines.extend([
        "",
        "---",
        "",
        "## Finalized Table: Mean Generation Time and Computational Complexity (GFLOPs)",
        "",
        "> **C++ Threader Process**: All thread synthesis generated strictly via native compiled `threader.exe`.",
        "> **Image Resolution**: Scaled long edge 640px.",
        "",
        "| Algorithm | Thread Count | Mean Gen Time (ms/img) | Throughput (img/sec) | Computational Complexity (GFLOPs) | Val mAP50 (% Base) | Test mAP50 (% Base) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    if bval and btest:
        md_lines.append(
            f"| **`Baseline (Original YOLOv12n)`** | **0 threads** | **0.0 ms** | **∞ fps** | **0.0000 GFLOPs** | "
            f"**{bval.get('map50', 0):.4f} (100.0%)** | **{btest.get('map50', 0):.4f} (100.0%)** |"
        )

    for k, v in progress.items():
        gen_ms = v.get("mean_gen_ms", 0.0)
        throughput = (1000.0 / gen_ms) if gen_ms > 0 else 0.0
        gflops = v.get("gflops", 0.0)
        vm = v["val"]
        tm = v["test"]
        bv_m50 = bval.get("map50", 1.0) or 1.0
        bt_m50 = btest.get("map50", 1.0) or 1.0
        norm_v_m50 = (vm.get("map50", 0.0) / bv_m50) * 100.0
        norm_t_m50 = (tm.get("map50", 0.0) / bt_m50) * 100.0
        md_lines.append(
            f"| **`{v['algo']}`** | {v['threads']:,} | {gen_ms:.1f} ms | {throughput:.1f} fps | {gflops:.4f} GFLOPs | "
            f"{vm.get('map50', 0):.4f} ({norm_v_m50:.1f}%) | {tm.get('map50', 0):.4f} ({norm_t_m50:.1f}%) |"
        )

    RESULTS_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

# ============================================================================
# Step 6: Benchmark Execution Engine (Supports arbitrary thread counts & phases)
# ============================================================================
def run_benchmark_for_thread_counts(thread_counts: list, phase_name: str):
    splits = get_or_create_splits()
    generate_yolo12n_ground_truth(splits)
    evaluate_baseline(splits)

    class_names = yolo_class_names()

    log(f"Starting {phase_name} across {len(ALGORITHMS)} algorithms x {len(thread_counts)} thread counts = {len(ALGORITHMS) * len(thread_counts)} runs...")

    for algo_name, algo_id in ALGORITHMS:
        log("\n=======================================================")
        log(f"[{phase_name}] Processing Algorithm [{algo_id}/10]: {algo_name}")
        log("=======================================================")

        algo_dir = THREADED_DATASET_DIR / algo_name
        algo_dir.mkdir(parents=True, exist_ok=True)

        for thread_count in thread_counts:
            run_key = f"{algo_name}_{thread_count}"

            # Always read fresh progress
            progress = {}
            if PROGRESS_FILE.exists():
                try:
                    with open(PROGRESS_FILE, "r") as f:
                        progress = json.load(f)
                except Exception:
                    pass

            if run_key in progress:
                log(f"Run {run_key} already completed, skipping.")
                continue

            free_gb = check_disk_space_gb()
            log(f"Free disk space: {free_gb:.1f} GB")

            total_paused_sec = 0.0
            total_paused_sec += wait_if_in_pause_window(f"{algo_name} | {thread_count} threads")

            run_start_time = time.time()
            subfolder = algo_dir / str(thread_count)
            img_train_dir = subfolder / "images" / "train"
            img_val_dir = subfolder / "images" / "val"
            img_test_dir = subfolder / "images" / "test"

            lbl_train_dir = subfolder / "labels" / "train"
            lbl_val_dir = subfolder / "labels" / "val"
            lbl_test_dir = subfolder / "labels" / "test"

            # Render threaded images via C++ threader.exe
            t_train = render_dataset_split(algo_id, algo_name, thread_count, "train", splits["train"], img_train_dir)
            t_val = render_dataset_split(algo_id, algo_name, thread_count, "val", splits["val"], img_val_dir)
            t_test = render_dataset_split(algo_id, algo_name, thread_count, "test", splits["test"], img_test_dir)

            all_durations = t_train + t_val + t_test
            if all_durations:
                mean_gen_ms = (sum(all_durations) / len(all_durations)) * 1000.0
            else:
                mean_gen_ms = benchmark_generation_time(algo_id, algo_name, thread_count)
            gflops = calculate_gflops(algo_id, thread_count)

            # Copy GT labels once
            lbl_train_dir.mkdir(parents=True, exist_ok=True)
            lbl_val_dir.mkdir(parents=True, exist_ok=True)
            lbl_test_dir.mkdir(parents=True, exist_ok=True)

            if len(list(lbl_train_dir.glob("*.txt"))) != len(splits["train"]):
                for fn in splits["train"]:
                    stem = Path(fn).stem
                    shutil.copy(GT_LABELS_DIR / f"{stem}.txt", lbl_train_dir / f"{stem}.txt")
            if len(list(lbl_val_dir.glob("*.txt"))) != len(splits["val"]):
                for fn in splits["val"]:
                    stem = Path(fn).stem
                    shutil.copy(GT_LABELS_DIR / f"{stem}.txt", lbl_val_dir / f"{stem}.txt")
            if len(list(lbl_test_dir.glob("*.txt"))) != len(splits["test"]):
                for fn in splits["test"]:
                    stem = Path(fn).stem
                    shutil.copy(GT_LABELS_DIR / f"{stem}.txt", lbl_test_dir / f"{stem}.txt")

            data_yaml = subfolder / "data.yaml"
            with open(data_yaml, "w") as f:
                yaml.dump({
                    "path": str(subfolder.resolve()).replace("\\", "/"),
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": class_names
                }, f)

            run_project = RUNS_DIR / algo_name
            run_name = f"threads_{thread_count}"
            last_weights = run_project / run_name / "weights" / "last.pt"

            # Training loop with automatic pause/resume support
            while True:
                total_paused_sec += wait_if_in_pause_window(f"{algo_name} | {thread_count} threads")
                try:
                    if last_weights.exists():
                        log(f"[{algo_name} | {thread_count} threads] Found existing checkpoint {last_weights}. Resuming training...")
                        model = YOLO(str(last_weights))
                        model.add_callback("on_fit_epoch_end", create_pause_callback(algo_name, thread_count))
                        model.train(resume=True)
                    else:
                        log(f"[{algo_name} | {thread_count} threads] Starting YOLOv12n training (100 epochs, batch 16, seed 1)...")
                        model = YOLO(str(MODELS_DIR / "yolo12n.pt"))
                        model.add_callback("on_fit_epoch_end", create_pause_callback(algo_name, thread_count))
                        model.train(
                            data=str(data_yaml),
                            epochs=100,
                            batch=16,
                            imgsz=640,
                            device=0,
                            seed=1,
                            workers=2,
                            save=True,
                            save_period=-1,
                            plots=False,
                            val=True,
                            project=str(run_project),
                            name=run_name,
                            verbose=False
                        )
                    break # Training completed 100 epochs
                except PauseTrainingInterrupt:
                    log(f"[{algo_name} | {thread_count} threads] Checkpoint preserved at {last_weights}. Releasing GPU memory...")
                    try:
                        del model
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
                    total_paused_sec += wait_if_in_pause_window(f"{algo_name} | {thread_count} threads")
                    # Loops back and resumes from last_weights!

            total_paused_sec += wait_if_in_pause_window(f"{algo_name} | {thread_count} threads")

            # Validation & Testing at tau = 0.3
            log(f"[{algo_name} | {thread_count} threads] Running validation and testing at tau = 0.3...")
            best_weights = run_project / run_name / "weights" / "best.pt"
            eval_model = YOLO(str(best_weights if best_weights.exists() else MODELS_DIR / "yolo12n.pt"))

            val_res = eval_model.val(data=str(data_yaml), split="val", conf=0.3, batch=16, device=0, verbose=False)
            test_res = eval_model.val(data=str(data_yaml), split="test", conf=0.3, batch=16, device=0, verbose=False)

            val_metrics = extract_metrics(val_res)
            test_metrics = extract_metrics(test_res)

            elapsed_min = max(0.0, (time.time() - run_start_time - total_paused_sec) / 60.0)
            log(f"[{algo_name} | {thread_count} threads] DONE in {elapsed_min:.1f} min (active time)! Mean Gen: {mean_gen_ms:.1f} ms, GFLOPs: {gflops:.4f}, Val mAP50={val_metrics['map50']:.4f}, Test mAP50={test_metrics['map50']:.4f}")

            # Append results dynamically
            append_result(algo_name, algo_id, thread_count, val_metrics, test_metrics, elapsed_min, mean_gen_ms, gflops)

            # Clean last.pt checkpoint to save disk space
            last_weights = run_project / run_name / "weights" / "last.pt"
            if last_weights.exists():
                try:
                    last_weights.unlink()
                except Exception:
                    pass

            # Delete that single complete run's images immediately, then continue onto the next
            img_dir = subfolder / "images"
            if img_dir.exists():
                log(f"[{algo_name} | {thread_count} threads] Run complete. Deleting images for {run_key} ({img_dir}) before continuing to the next...")
                shutil.rmtree(img_dir, ignore_errors=True)


def main():
    global PAUSE_ENABLED
    parser = argparse.ArgumentParser(description="Digital Filography benchmark")
    parser.add_argument("--daily-pause", action="store_true",
                        help="pause GPU work between 15:00 and 23:00 each day")
    PAUSE_ENABLED = parser.parse_args().daily_pause
    if not THREADER_EXE.exists():
        raise SystemExit(f"Missing {THREADER_EXE}. Run native\\build.bat first.")

    # Phase 1: Scheduled benchmark runs (1,000 to 10,000 threads for all 10 algorithms)
    log("================================================================================")
    log("PHASE 1: Running scheduled benchmark (1,000 to 10,000 threads for all 10 algorithms)")
    log("================================================================================")
    run_benchmark_for_thread_counts(THREAD_COUNTS, "Phase 1 (1k..10k threads)")

    log("\n================================================================================")
    log("PHASE 1 COMPLETE! All 10 algorithms finished 1,000 to 10,000 threads.")
    log("================================================================================")

    # Phase 2: Extended benchmark runs (11,000 to 15,000 threads for all 10 algorithms)
    log("\n================================================================================")
    log("PHASE 2: Running extended benchmark (11,000 to 15,000 threads for all 10 algorithms)")
    log("================================================================================")
    run_benchmark_for_thread_counts(EXTENDED_THREAD_COUNTS, "Phase 2 (11k..15k threads)")

    log("\n================================================================================")
    log("ALL BENCHMARK PHASES (1,000 to 15,000 THREADS FOR ALL ALGORITHMS) COMPLETED!")
    log("================================================================================")


if __name__ == "__main__":
    main()
