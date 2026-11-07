# Source Files

The source package is organized into three modular layers:
1. **[`config_manager.py`](config_manager.py)**: Central configuration, hardware/CUDA resolution, random seed management, settings schema, and App state.
2. **[`eval_engine.py`](eval_engine.py)**: Computational pipeline engine for dataset prep, latency benchmarks, multi-seed validation loops with Mean ± Standard Deviation statistics, and training runs.
3. **[`run_terminal.py`](run_terminal.py)**: Menu-driven terminal UI, keyboard navigation, and CLI runner.

---

## `run_threaded_experiment.py`
The filography benchmark: splits the dataset, makes YOLOv12n pseudo labels, renders
each algorithm and thread count with `build/threader.exe`, trains, and writes
`results/threaded_benchmark_results.*`. Runs resume from
`results/threaded_benchmark_progress.json`.

---

## `run_terminal.py`
Interactive menu-driven terminal for configuring and launching dataset downloads,
YOLO model selection, and CUDA hardware benchmarking; run it from the project root with `python src/run_terminal.py`.

It can also be invoked directly with flags:
- `python src/run_terminal.py --run-tests`: Run multi-seed validation on all active models non-interactively.
- `python src/run_terminal.py --clear-results`: Clear previous run outputs without menu prompts.

### Screens
| Screen | Purpose |
|---|---|
| Download dataset | Pick COCO classes, set image count, start download |
| Clear dataset | Delete images from a specific class folder (`dataset/<class>/`) or all folders |
| Select model(s) | Choose one or multiple YOLO models (`yolo26n.pt`, `yolo11n.pt`, `yolov8n.pt`, `yolov10n.pt`, `yolo12n.pt`) |
| Configure shared settings | Manage global pipeline and model settings: compute device (`CUDA:0`), confidence threshold (`conf`), NMS threshold (`iou`), FP16 half-precision (`half`), max detections (`max_det`), evaluation plots (`plots`), COCO JSON export (`save_json`), class-agnostic NMS (`agnostic_nms`), test-time augmentation (`augment`), dataset split (`split`), random seed(s), epochs, batch size, learning rate, image size, workers |
| Configure tracked metrics | Toggle metrics tracking (speed breakdown, model complexity, GPU VRAM, accuracy) and benchmark iterations |
| Test latency (benchmark) | Run real-time inference latency benchmark (avg ms, FPS, min/max, VRAM, speed breakdown) on CUDA GPU or CPU |
| Run tests (evaluation) | Evaluate object detection performance across all specified seeds calculating Mean ± Standard Deviation (mAP50, mAP50-95, P, R, F1, speed) directly on dataset |
| Run training | Review active models, shared settings, and initiate training pipeline |
| Clear results | Remove previous run artefacts (`runs/`, `results/`) |
| All settings | View and edit all shared pipeline and metric settings from a single place |

---

## `eval_engine.py`
Core computational workflows:
- **Multi-Seed Testing**: Iterates through each configured seed, executes `model.val()`, and computes sample Mean ± Standard Deviation for mAP50, mAP50-95, precision, recall, F1-score, and latency.
- **Latency Benchmarking**: Measures warmup and timed passes on CUDA GPU or CPU with speed breakdown and VRAM tracking.
- **Dataset Preparation**: Generates YOLO annotation `.txt` files directly from COCO annotations and creates runtime `test_data.yaml`.

---

## `config_manager.py`
- Stores configuration schemas (`SHARED_SETTINGS_SCHEMA`, `METRICS_SCHEMA`).
- Validates and coerces user inputs with `parse_value` and formats display with `format_value`.
- Enforces CUDA GPU detection and device string normalization (`resolve_device`).
- Manages `App` state and persistence to `config/config.json`.
Pretrained weights located in `models/`:
- `yolo26n.pt`
- `yolo11n.pt`
- `yolov8n.pt`
- `yolov10n.pt`
- `yolo12n.pt`

Users can select a single model or toggle multiple models simultaneously from the terminal.

---

## Configuration (`config/config.json`)
Settings are organized into `selected_models`, `shared` pipeline settings, and `metrics` tracking preferences:

```json
{
  "selected_models": [
    "yolov8n.pt",
    "yolo11n.pt"
  ],
  "shared": {
    "device": "CUDA:0",
    "conf": 0.25,
    "iou": 0.70,
    "half": false,
    "max_det": 300,
    "plots": true,
    "save_json": false,
    "agnostic_nms": false,
    "augment": false,
    "split": "val",
    "seeds": [0],
    "epochs": 10,
    "batch_size": 32,
    "imgsz": 640,
    "learning_rate": 0.001,
    "workers": 8,
    "num_images": 5000,
    "selected_classes": []
  },
  "metrics": {
    "speed_breakdown": true,
    "model_complexity": true,
    "vram_usage": true,
    "detection_accuracy": true,
    "iterations": 20,
    "warmup_passes": 5
  }
}
```

### CUDA:0 Hardware Handling
When `CUDA:0` is configured:
1. `resolve_device()` checks `torch.cuda.is_available()`.
2. Sets `torch.cuda.set_device(0)`.
3. Normalizes device string to lowercase `cuda:0` required by PyTorch and Ultralytics.
4. Verifies model tensor placement directly on the GPU.

---

## `download_dataset.py`
Downloads COCO val2017 images into per-class subfolders and writes a filtered
`annotations.json` inside each one; reads defaults (`num_images`, `selected_classes`, and `workers`) from `config/config.json`.

**CLI usage**
```bash
# Basic — downloads images into dataset/ (reads count and classes from config/config.json)
python src/download_dataset.py

# Fixed count, no class filter
python src/download_dataset.py -n 500

# Per-class subfolders — 100 images each into dataset/dog/, dataset/person/, dataset/cat/
python src/download_dataset.py -n 300 --classes dog person cat

# List all 80 valid class names
python src/download_dataset.py --list-classes
```

**Disk layout produced by `--classes`**
```
dataset/
  dog/
    000001.jpg
    annotations.json   ← COCO-format, filtered to dog only
  person/
    000002.jpg
    annotations.json
```

---

## Results Directory (`results/<timeofstarting>/`)
Whenever you execute a benchmark, test evaluation, or training run, results are automatically organized into:
```
results/
  YYYY-MM-DD_HH-MM-SS/
    latency_benchmark.json    ← Machine-readable benchmark metrics
    latency_benchmark.txt     ← Formatted benchmark summary table
    test_results.json         ← Machine-readable model test metrics (mAP50, mAP50-95, P, R, F1, speed)
    test_summary.txt          ← Formatted evaluation comparison table
    test_data.yaml            ← Generated dataset YAML configuration for testing
    val_images.txt            ← Manifest of dataset images evaluated
Previous runs can be selectively viewed and deleted through the **Clear results** menu in `run_terminal.py`.

---

## `generate_report.py` (Research Reporting & Synthesis)
Compiles benchmark and evaluation outputs into comprehensive, publication-grade research reports.

**Key capabilities**:
- **Data Aggregation**: Ingests `test_results.json`, calculating sample Mean ± Standard Deviation ($s = \sqrt{\frac{1}{N-1}\sum (x_i - \bar{x})^2}$) across seeds.
- **Diagnostic Asset Archival**: Automatically extracts and copies key model visual figures (Normalized Confusion Matrices, Precision-Recall curves, F1-confidence curves, sample prediction batches) into `reports/assets/<run_id>/` so reports remain fully intact after clearing raw run scratch.
- **Pareto & Efficiency Analysis**: Evaluates parameter efficiency (mAP50 / MParams), latency breakdowns (preprocess, forward, postprocess), and throughput.
- **Hypothesis & Variance Analysis**: Documents why fixed-weight validation without TTA exhibits zero metric variance across seeds vs. physical hardware timing jitter.

**CLI Usage**:
```bash
# Generate report for the latest benchmark run and archive visual assets
python src/generate_report.py

# Generate report and automatically clear the results/ folder afterwards
python src/generate_report.py --clear-results

# Target a specific run ID
python src/generate_report.py --run-id 2026-10-14_20-30-07
```

---

## Reports Directory (`reports/`)
Permanent repository for publication-ready experiment reports:
```
reports/
  evaluation_report_YYYY-MM-DD_HH-MM-SS.md   ← Timestamped research report
  latest_report.md                           ← Pointer to the most recent evaluation report
  assets/
    YYYY-MM-DD_HH-MM-SS/
      <model_name>/
        confusion_matrix_normalized.png      ← Archived model confusion matrix
        BoxPR_curve.png                      ← Archived precision-recall curve
        BoxF1_curve.png                      ← Archived F1-confidence curve
        val_batch0_pred.jpg                  ← Archived sample detection batch
```


