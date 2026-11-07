#!/usr/bin/env python3
"""Core configuration, hardware management, and settings schema for Filography.

Provides:
- Filesystem paths and model/class catalogues
- CUDA and compute device resolution
- Random seed initialization for reproducible experiments
- Shared pipeline and metrics schemas
- Value parsing, formatting, loading, and persistence
- Centralized App state management
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT      = Path(__file__).resolve().parent.parent
CONFIG_PATH       = PROJECT_ROOT / "config" / "config.json"
DATASET_DIR       = PROJECT_ROOT / "dataset"
MODELS_DIR        = PROJECT_ROOT / "models"
RESULTS_DIR       = PROJECT_ROOT / "results"
RUNS_DIR          = PROJECT_ROOT / "runs"
ANNOTATIONS_CACHE = PROJECT_ROOT / "config" / "instances_val2017.json"

# ---------------------------------------------------------------------------
# YOLO Models Catalogue
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = (
    "yolo26n.pt",
    "yolo11n.pt",
    "yolov8n.pt",
    "yolov10n.pt",
    "yolo12n.pt",
)


def get_available_models() -> list[str]:
    """Return available models in models/ directory, preserving catalog order."""
    found = set()
    if MODELS_DIR.exists():
        found = {f.name for f in MODELS_DIR.glob("*.pt")}
    catalog = [m for m in AVAILABLE_MODELS if m in found or not found]
    extra = sorted(found - set(AVAILABLE_MODELS))
    return catalog + extra


# ---------------------------------------------------------------------------
# COCO class catalogue (val2017 - 80 categories, alphabetical)
# ---------------------------------------------------------------------------

COCO_CLASSES = (
    "airplane",      "apple",         "backpack",     "banana",        "baseball bat",
    "baseball glove","bear",           "bed",          "bench",         "bicycle",
    "bird",          "boat",          "book",          "bottle",        "bowl",
    "broccoli",      "bus",           "cake",          "car",           "carrot",
    "cat",           "cell phone",    "chair",         "clock",         "couch",
    "cow",           "cup",           "dining table",  "dog",           "donut",
    "elephant",      "fire hydrant",  "fork",          "frisbee",       "giraffe",
    "hair drier",    "handbag",       "horse",         "hot dog",       "keyboard",
    "kite",          "knife",         "laptop",        "microwave",     "motorcycle",
    "mouse",         "orange",        "oven",          "parking meter", "person",
    "pizza",         "potted plant",  "refrigerator",  "remote",        "sandwich",
    "scissors",      "sheep",         "sink",          "skateboard",    "skis",
    "snowboard",     "sports ball",   "spoon",         "stop sign",     "suitcase",
    "surfboard",     "teddy bear",    "tennis racket", "tie",           "toaster",
    "toilet",        "toothbrush",    "traffic light", "train",         "truck",
    "tv",            "umbrella",      "vase",          "wine glass",    "zebra",
)

# ---------------------------------------------------------------------------
# CUDA / Device Resolution & Reproducibility
# ---------------------------------------------------------------------------

def get_cuda_status() -> tuple[bool, str, str | None]:
    """Return (is_available, description, device_name)."""
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            return True, f"CUDA available ({count} GPU(s))", name
        return False, "CUDA not available (CPU only)", None
    except ImportError:
        return False, "PyTorch not installed", None
    except Exception as exc:  # noqa: BLE001
        return False, f"CUDA error: {exc}", None


def resolve_device(device_str: str = "CUDA:0") -> str:
    """Normalize device string and ensure CUDA:0 resolves to lowercase cuda device."""
    raw = str(device_str).strip()
    lower = raw.lower()
    if lower.startswith("cuda") or lower.isdigit():
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required to use CUDA device.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device '{device_str}' requested, but CUDA is not available in PyTorch. "
                "Please verify NVIDIA drivers and CUDA PyTorch installation."
            )
        if lower.isdigit():
            idx = int(lower)
        else:
            parts = lower.split(":")
            idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {idx} requested ('{device_str}'), but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )
        torch.cuda.set_device(idx)
        return f"cuda:{idx}"
    elif lower == "cpu":
        return "cpu"
    return lower


def set_seed(seed: int = 0):
    """Set random seed across Python random, NumPy, and PyTorch for reproducible runs."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Settings Schema
# ---------------------------------------------------------------------------

SHARED_SETTINGS_SCHEMA = {
    "device": {
        "group":   "Hardware",
        "label":   "Compute device",
        "type":    str,
        "default": "CUDA:0",
    },
    "conf": {
        "group":   "Testing & Inference",
        "label":   "Confidence threshold (conf)",
        "type":    float,
        "default": 0.25,
        "min":     0.01,
        "max":     1.0,
    },
    "iou": {
        "group":   "Testing & Inference",
        "label":   "NMS IoU threshold (iou)",
        "type":    float,
        "default": 0.70,
        "min":     0.01,
        "max":     1.0,
    },
    "half": {
        "group":   "Testing & Inference",
        "label":   "FP16 half-precision (half)",
        "type":    bool,
        "default": False,
    },
    "max_det": {
        "group":   "Testing & Inference",
        "label":   "Max detections per image (max_det)",
        "type":    int,
        "default": 300,
        "min":     1,
        "max":     1000,
    },
    "plots": {
        "group":   "Testing & Inference",
        "label":   "Save evaluation plots (plots)",
        "type":    bool,
        "default": True,
    },
    "save_json": {
        "group":   "Testing & Inference",
        "label":   "Save COCO format JSON (save_json)",
        "type":    bool,
        "default": False,
    },
    "agnostic_nms": {
        "group":   "Testing & Inference",
        "label":   "Class-agnostic NMS (agnostic_nms)",
        "type":    bool,
        "default": False,
    },
    "augment": {
        "group":   "Testing & Inference",
        "label":   "Test-time augmentation (augment)",
        "type":    bool,
        "default": False,
    },
    "split": {
        "group":   "Testing & Inference",
        "label":   "Dataset split (split)",
        "type":    str,
        "default": "val",
    },
    "epochs": {
        "group":   "Training",
        "label":   "Epochs",
        "type":    int,
        "default": 10,
        "min":     1,
        "max":     200,
    },
    "batch_size": {
        "group":   "Training",
        "label":   "Batch size",
        "type":    int,
        "default": 32,
        "min":     1,
        "max":     128,
    },
    "learning_rate": {
        "group":   "Training",
        "label":   "Learning rate",
        "type":    float,
        "default": 0.001,
        "min":     0.000001,
        "max":     1.0,
    },
    "imgsz": {
        "group":   "Pipeline",
        "label":   "Image size (imgsz)",
        "type":    int,
        "default": 640,
        "min":     128,
        "max":     1280,
    },
    "workers": {
        "group":   "Pipeline",
        "label":   "Worker threads",
        "type":    int,
        "default": 8,
        "min":     1,
        "max":     32,
    },
    "seeds": {
        "group":   "Pipeline",
        "label":   "Random seed(s)",
        "type":    list,
        "default": [0],
    },
    "num_images": {
        "group":   "Dataset",
        "label":   "Number of images",
        "type":    int,
        "default": 5000,
        "min":     1,
        "max":     5000,
    },
}

METRICS_SCHEMA = {
    "speed_breakdown": {
        "label":       "Speed breakdown (preprocess, inference, postprocess)",
        "description": "Measure separate timings for image prep, model forward, and NMS in ms",
        "type":        bool,
        "default":     True,
    },
    "model_complexity": {
        "label":       "Model complexity (parameter count & file size)",
        "description": "Extract total parameters (M) and model weights disk size (MB)",
        "type":        bool,
        "default":     True,
    },
    "vram_usage": {
        "label":       "GPU VRAM memory tracking (allocated & reserved)",
        "description": "Track peak CUDA memory allocated and reserved during inference (MB)",
        "type":        bool,
        "default":     True,
    },
    "detection_accuracy": {
        "label":       "Detection quality metrics (mAP50, mAP50-95, P, R)",
        "description": "Evaluate object detection precision, recall, and mAP on dataset",
        "type":        bool,
        "default":     True,
    },
    "iterations": {
        "label":       "Benchmark iterations",
        "description": "Number of timed iterations per model for latency / FPS benchmarking",
        "type":        int,
        "default":     20,
        "min":         5,
        "max":         500,
    },
    "warmup_passes": {
        "label":       "Warmup passes",
        "description": "Number of warmup inference passes before measuring latency",
        "type":        int,
        "default":     5,
        "min":         0,
        "max":         50,
    },
}


def format_value(spec, value):
    """Format setting value for display."""
    if spec["type"] is bool:
        return "True" if value else "False"
    if spec["type"] is list:
        if isinstance(value, list):
            if not value:
                return "none"
            if len(value) == 1:
                return str(value[0])
            if len(value) <= 3:
                return f"{', '.join(str(v) for v in value)} ({len(value)} seeds)"
            return f"{', '.join(str(v) for v in value[:3])} (+{len(value) - 3} more)"
        return str(value)
    if spec["type"] is int:
        return f"{value:,}"
    if spec["type"] is float:
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def parse_value(spec, text):
    """Convert text input into a validated value according to schema. Returns (value, error_message)."""
    if spec["type"] is list:
        if isinstance(text, list):
            vals = [int(x) for x in text if str(x).replace("_", "").isdigit() and int(str(x).replace("_", "")) >= 0]
            return vals or [0], None
        cleaned = str(text).strip().strip("[]").replace(",", " ").split()
        vals = []
        for c in cleaned:
            c_clean = c.replace("_", "")
            if c_clean.isdigit() and int(c_clean) >= 0:
                vals.append(int(c_clean))
            else:
                return None, f"'{c}' is not a valid non-negative integer."
        return vals or [0], None

    text = str(text).strip()
    if spec["type"] is bool:
        lower = text.lower()
        if lower in ("true", "t", "1", "yes", "y", "on"):
            return True, None
        if lower in ("false", "f", "0", "no", "n", "off"):
            return False, None
        return None, "Please enter true/false (or y/n)."

    if spec["type"] is str:
        if not text:
            return None, "Value cannot be empty."
        return text, None

    clean = text.replace("_", "")
    if spec["type"] is int:
        clean = clean.replace(",", "")
    try:
        value = spec["type"](clean)
    except ValueError:
        kind = "a whole number" if spec["type"] is int else "a number"
        return None, f"Please enter {kind}."

    if "min" in spec and "max" in spec and not (spec["min"] <= value <= spec["max"]):
        low  = format_value(spec, spec["min"])
        high = format_value(spec, spec["max"])
        return None, f"The value must be between {low} and {high}."
    return value, None


# ---------------------------------------------------------------------------
# Config Load and Save
# ---------------------------------------------------------------------------

def load_settings():
    """Load settings from config/config.json."""
    shared = {k: spec["default"] for k, spec in SHARED_SETTINGS_SCHEMA.items()}
    shared["selected_classes"] = []
    selected_models = ["yolov8n.pt"]
    metrics = {k: spec["default"] for k, spec in METRICS_SCHEMA.items()}

    try:
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {
            "shared": shared,
            "selected_models": selected_models,
            "metrics": metrics,
        }

    if not isinstance(saved, dict):
        return {
            "shared": shared,
            "selected_models": selected_models,
            "metrics": metrics,
        }

    # 1. Load shared settings
    shared_src = saved.get("shared", saved)  # fallback to top-level for legacy config
    if isinstance(shared_src, dict):
        for key, spec in SHARED_SETTINGS_SCHEMA.items():
            if key in shared_src:
                val, err = parse_value(spec, shared_src[key] if isinstance(shared_src[key], list) else str(shared_src[key]))
                if err is None:
                    shared[key] = val
        # Sync seeds and seed
        raw_seeds = shared_src.get("seeds")
        if isinstance(raw_seeds, list) and raw_seeds:
            s_parsed = [int(s) for s in raw_seeds if str(s).replace("_", "").isdigit() and int(str(s).replace("_", "")) >= 0]
            if s_parsed:
                shared["seeds"] = s_parsed
                shared["seed"] = s_parsed[0]
        elif "seed" in shared_src:
            try:
                s_val = int(shared_src["seed"])
                if s_val >= 0:
                    shared["seeds"] = [s_val]
                    shared["seed"] = s_val
            except (ValueError, TypeError):
                pass
        raw_classes = shared_src.get("selected_classes", saved.get("selected_classes", []))
        if isinstance(raw_classes, list):
            shared["selected_classes"] = [c for c in raw_classes if isinstance(c, str) and c in COCO_CLASSES]

    # 2. Load selected models
    raw_models = saved.get("selected_models")
    if isinstance(raw_models, list) and raw_models:
        valid = [m for m in raw_models if isinstance(m, str)]
        if valid:
            selected_models = valid

    # 3. Load metrics settings
    metrics_src = saved.get("metrics")
    if isinstance(metrics_src, dict):
        for k, spec in METRICS_SCHEMA.items():
            if k in metrics_src:
                val, err = parse_value(spec, str(metrics_src[k]))
                if err is None:
                    metrics[k] = val

    return {
        "shared": shared,
        "selected_models": selected_models,
        "metrics": metrics,
    }


def save_settings(data):
    """Write settings dict to config/config.json. Returns True on success."""
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Ensure seeds and seed stay in sync
        shared_seeds = data["shared"].get("seeds", [0])
        if not isinstance(shared_seeds, list) or not shared_seeds:
            shared_seeds = [data["shared"].get("seed", 0)]
        data["shared"]["seeds"] = shared_seeds
        data["shared"]["seed"] = shared_seeds[0]

        # Construct clean, organized JSON structure
        payload = {
            "selected_models": data["selected_models"],
            "shared": data["shared"],
            "metrics": data.get("metrics", {k: s["default"] for k, s in METRICS_SCHEMA.items()}),
        }
        CONFIG_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# App State
# ---------------------------------------------------------------------------

class App:
    """Holds global application state, active settings, and navigation stack."""

    def __init__(self):
        loaded                = load_settings()
        self.shared_settings  = loaded["shared"]
        self.selected_models  = list(loaded["selected_models"])
        self.metrics          = loaded.get("metrics", {k: s["default"] for k, s in METRICS_SCHEMA.items()})
        self.selected_classes = list(self.shared_settings.get("selected_classes", []))
        self.current          = "main"
        self.back_stack       = []
        self.forward_stack    = []
        self.message          = None
        self.running          = True

    def save(self):
        """Persist settings to config/config.json."""
        self.shared_settings["selected_classes"] = self.selected_classes
        payload = {
            "selected_models": self.selected_models,
            "shared": self.shared_settings,
            "metrics": self.metrics,
        }
        return save_settings(payload)

    def notify(self, kind, text):
        """Queue a notification message for next render."""
        self.message = (kind, text)

    def go(self, screen_id):
        if screen_id == self.current:
            return
        self.back_stack.append(self.current)
        self.forward_stack.clear()
        self.current = screen_id

    def back(self):
        if not self.back_stack:
            self.notify("info", "There is nothing to go back to.")
            return
        self.forward_stack.append(self.current)
        self.current = self.back_stack.pop()

    def forward(self):
        if not self.forward_stack:
            self.notify("info", "There is nothing to go forward to.")
            return
        self.back_stack.append(self.current)
        self.current = self.forward_stack.pop()
