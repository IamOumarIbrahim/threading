#!/usr/bin/env python3
"""Interactive menu-driven terminal for dataset management and YOLO model workflows.

Start it from the project root:
    python src/run_terminal.py

Non-interactive commands:
    python src/run_terminal.py --run-tests
    python src/run_terminal.py --clear-results

Keys that work on every screen:
    1, 2, 3 ...  choose an option
    b            go back (like a browser)
    f            go forward
    h            go to the main menu
    q            quit
"""

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Add src to module path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_manager import (
    COCO_CLASSES,
    METRICS_SCHEMA,
    MODELS_DIR,
    SHARED_SETTINGS_SCHEMA,
    App,
    format_value,
    get_available_models,
    get_cuda_status,
    parse_value,
)
from eval_engine import (
    _count_all_images,
    _list_result_runs,
    action_clear_dataset,
    action_clear_results,
    action_run_tests,
    action_run_training,
    action_select_all_models,
    action_select_single_model,
    action_start_download,
    action_test_latency,
    action_toggle_model,
)

# ---------------------------------------------------------------------------
# Terminal setup & ANSI Styling
# ---------------------------------------------------------------------------

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

if os.name == "nt":
    os.system("")   # enable ANSI colors in the Windows console

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"
GRAY    = "\033[90m"


def paint(text, *styles):
    """Wrap text in color/style codes (does nothing if colors are off)."""
    if not USE_COLOR or not styles:
        return str(text)
    return "".join(styles) + str(text) + RESET


def clear_screen():
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")


def banner(title, color=CYAN):
    """Print a title inside a box."""
    width = max(len(title) + 8, 36)
    print(paint("╔" + "═" * (width - 2) + "╗", BOLD, color))
    print(paint("║", BOLD, color) + paint(title.center(width - 2), BOLD, WHITE) + paint("║", BOLD, color))
    print(paint("╚" + "═" * (width - 2) + "╝", BOLD, color))


MESSAGE_STYLES = {
    "success": ("✔", GREEN),
    "info":    ("ℹ", CYAN),
    "warning": ("⚠", YELLOW),
    "error":   ("✖", RED),
}


def say(kind, text):
    """Print a message with a colored symbol. Kind: success, info, warning, error."""
    symbol, color = MESSAGE_STYLES.get(kind, ("•", WHITE))
    print(paint(f"{symbol} ", BOLD, color) + text)


def confirm(question, default=False):
    """Ask a yes/no question."""
    hint   = "Y/n" if default else "y/N"
    answer = input(paint("? ", BOLD, MAGENTA) + f"{question} ({hint}) ").strip().lower()
    if not answer:
        return default
    return answer.startswith("y")


# ---------------------------------------------------------------------------
# UI Element Classes
# ---------------------------------------------------------------------------

@dataclass
class Item:
    label:  str
    action: Callable
    detail: str = ""


@dataclass
class Screen:
    title: str
    items: Callable
    body:  Callable = lambda app: []
    color: str = CYAN


# ---------------------------------------------------------------------------
# Shared Settings Interactive Prompts
# ---------------------------------------------------------------------------

def prompt_multiple_seeds(current_seeds: list[int]) -> list[int] | None:
    """Interactive CLI to prompt for number of seeds and specific seed values."""
    print()
    print(paint("Random Seeds Configuration (Multi-Seed Testing)", BOLD, CYAN))
    current_str = ", ".join(str(s) for s in current_seeds) if current_seeds else "none"
    print(f"Current seed(s): {paint(current_str, BOLD, GREEN)} ({len(current_seeds)} active)")
    print()

    while True:
        raw_count = input(paint("Enter how many seeds you want (press Enter to cancel): ", BOLD, MAGENTA)).strip()
        if not raw_count:
            return None
        if not raw_count.isdigit() or int(raw_count) < 1:
            say("warning", "Please enter a whole number of at least 1 (e.g. 1, 2, 5).")
            continue
        count = int(raw_count)
        break

    print()
    if count == 1:
        print("Please enter the seed to use (whole number >= 0):")
    else:
        print(f"Please enter the {count} seeds that you want to use (e.g. 1, 49, 84, 82, 9):")
        print(paint("Hint: You can enter them one by one, or all at once separated by commas/spaces.", DIM))
    print()

    seeds: list[int] = []
    while len(seeds) < count:
        count - len(seeds)
        idx = len(seeds) + 1
        prompt_txt = f"Seed #{idx} of {count}: " if count > 1 else "Seed: "
        raw_val = input(paint(prompt_txt, BOLD, MAGENTA)).strip()
        if not raw_val:
            if seeds:
                print(paint(f"Using {len(seeds)} seed(s): {', '.join(str(s) for s in seeds)}", CYAN))
                return seeds
            return None

        # Check for multiple values on one line
        tokens = raw_val.replace(",", " ").split()
        if len(tokens) > 1:
            batch_vals = []
            for tok in tokens:
                clean = tok.replace("_", "")
                if clean.isdigit() and int(clean) >= 0:
                    batch_vals.append(int(clean))
                else:
                    say("warning", f"Skipping invalid seed '{tok}' (must be whole number >= 0).")
            for v in batch_vals:
                if len(seeds) < count:
                    seeds.append(v)
            if len(seeds) < count:
                print(paint(f"Added {len(batch_vals)} seeds ({', '.join(str(s) for s in seeds)}). Still need {count - len(seeds)} more.", CYAN))
            continue

        clean = raw_val.replace("_", "")
        if not clean.isdigit() or int(clean) < 0:
            say("warning", "Seed must be a non-negative whole number (e.g. 0, 42, 100).")
            continue

        seeds.append(int(clean))

    return seeds


def edit_shared_setting(app, key: str):
    spec = SHARED_SETTINGS_SCHEMA[key]
    current = app.shared_settings.get(key, spec["default"])
    print()
    print(paint(spec["label"], BOLD, CYAN))

    if key in ("seeds", "seed"):
        current_seeds = app.shared_settings.get("seeds", [app.shared_settings.get("seed", 0)])
        new_seeds = prompt_multiple_seeds(current_seeds)
        if new_seeds is None:
            app.notify("info", "No changes made.")
            return
        app.shared_settings["seeds"] = new_seeds
        app.shared_settings["seed"] = new_seeds[0]
        seeds_str = ", ".join(str(s) for s in new_seeds)
        app.save()
        app.notify("success", f"Random seed(s) set to: {seeds_str} ({len(new_seeds)} active).")
        return

    if spec["type"] is bool:
        app.shared_settings[key] = not current
        app.save()
        app.notify("success", f"{spec['label']} toggled to {app.shared_settings[key]}.")
        return

    low  = format_value(spec, spec["min"]) if "min" in spec else None
    high = format_value(spec, spec["max"]) if "max" in spec else None
    cur_fmt = format_value(spec, current)
    if low is not None and high is not None:
        print(f"Allowed range: {low} to {high}. Current value: {cur_fmt}")
    else:
        print(f"Current value: {cur_fmt}")

    while True:
        raw = input(paint("New value (press Enter to cancel): ", BOLD, MAGENTA))
        if not raw.strip():
            app.notify("info", "No changes made.")
            return
        value, error = parse_value(spec, raw)
        if error:
            say("warning", error)
            continue
        app.shared_settings[key] = value
        app.save()
        app.notify("success", f"{spec['label']} set to {format_value(spec, value)}.")
        return


def shared_setting_item(app, key: str, with_group: bool = False):
    spec = SHARED_SETTINGS_SCHEMA[key]
    label = f"{spec['group']}: {spec['label']}" if with_group else f"Set {spec['label'].lower()}"
    val = app.shared_settings.get(key, spec["default"])
    detail = format_value(spec, val)
    if key == "device" and str(val).upper().startswith("CUDA"):
        cuda_ok, _, gpu_name = get_cuda_status()
        if cuda_ok and gpu_name:
            detail = f"{val} ({gpu_name})"
    return Item(
        label=label,
        action=lambda a, k=key: edit_shared_setting(a, k),
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Metrics Settings Interactive Prompts
# ---------------------------------------------------------------------------

def action_toggle_metric(app, key: str):
    spec = METRICS_SCHEMA[key]
    if spec["type"] is bool:
        app.metrics[key] = not app.metrics.get(key, spec["default"])
        app.save()
        status_txt = "ON" if app.metrics[key] else "OFF"
        app.notify("success", f"Metric '{spec['label']}' is now {status_txt}.")
    else:
        edit_metric_setting(app, key)


def edit_metric_setting(app, key: str):
    spec = METRICS_SCHEMA[key]
    current = app.metrics.get(key, spec["default"])
    print()
    print(paint(spec["label"], BOLD, CYAN))
    if "description" in spec:
        print(paint(spec["description"], DIM))
    low = format_value(spec, spec["min"])
    high = format_value(spec, spec["max"])
    print(f"Allowed range: {low} to {high}. Current value: {current}")
    raw = input(paint("New value (press Enter to cancel): ", BOLD, MAGENTA))
    if not raw.strip():
        app.notify("info", "No changes made.")
        return
    value, error = parse_value(spec, raw)
    if error:
        say("warning", error)
        return
    app.metrics[key] = value
    app.save()
    app.notify("success", f"{spec['label']} set to {value}.")


def action_enable_all_metrics(app):
    for k, spec in METRICS_SCHEMA.items():
        if spec["type"] is bool:
            app.metrics[k] = True
    app.save()
    app.notify("success", "Enabled all metrics tracking.")


def action_disable_all_metrics(app):
    for k, spec in METRICS_SCHEMA.items():
        if spec["type"] is bool:
            app.metrics[k] = False
    app.save()
    app.notify("info", "Disabled all optional metrics tracking.")


def action_reset_metrics(app):
    for k, spec in METRICS_SCHEMA.items():
        app.metrics[k] = spec["default"]
    app.save()
    app.notify("success", "Reset tracked metrics to default.")


def metrics_items(app):
    items = []
    for key, spec in METRICS_SCHEMA.items():
        val = app.metrics.get(key, spec["default"])
        if spec["type"] is bool:
            detail = paint("✔ ON", GREEN) if val else paint("✘ OFF", GRAY)
        else:
            detail = str(val)
        items.append(
            Item(
                label=spec["label"],
                action=lambda a, k=key: action_toggle_metric(a, k),
                detail=detail,
            )
        )
    items.append(Item("Enable all metrics", action_enable_all_metrics))
    items.append(Item("Disable all metrics", action_disable_all_metrics))
    items.append(Item("Reset metrics to defaults", action_reset_metrics))
    return items


def metrics_body(app):
    active_count = sum(1 for k, spec in METRICS_SCHEMA.items() if spec["type"] is bool and app.metrics.get(k, spec["default"]))
    return [
        "Select an option to toggle tracking on/off or configure timed benchmark parameters.",
        f"Active metrics tracking: {paint(f'{active_count}/4 active', BOLD, GREEN)}",
        f"Benchmark passes: {paint(app.metrics.get('iterations', 20), BOLD, YELLOW)} iterations, {paint(app.metrics.get('warmup_passes', 5), BOLD, YELLOW)} warmup",
    ]


# ---------------------------------------------------------------------------
# Class Selection
# ---------------------------------------------------------------------------

def action_toggle_class(app, class_name: str):
    if class_name in app.selected_classes:
        app.selected_classes.remove(class_name)
    else:
        app.selected_classes.append(class_name)
    app.save()


def action_select_all_classes(app):
    app.selected_classes = list(COCO_CLASSES)
    app.save()
    app.notify("success", "Selected all 80 COCO classes.")


def action_clear_classes(app):
    app.selected_classes = []
    app.save()
    app.notify("info", "Cleared class selection. Will download by image ID.")


def class_items(app):
    items = []
    for name in COCO_CLASSES:
        tag = "✔" if name in app.selected_classes else ""
        items.append(
            Item(
                label=name,
                action=lambda a, c=name: action_toggle_class(a, c),
                detail=tag,
            )
        )
    items.append(Item("Select all 80 classes", action_select_all_classes))
    items.append(Item("Clear selection (all images)", action_clear_classes))
    return items


def class_body(app):
    n = len(app.selected_classes)
    if not n:
        return [
            "Pick a number to toggle  ✔ on/off.",
            f"Active: {paint('None (all images)', BOLD, YELLOW)} — downloads by ID into dataset/",
        ]
    shown = ", ".join(app.selected_classes[:5])
    if n > 5:
        shown += f" +{n - 5} more"
    return [
        "Pick a number to toggle  ✔ on/off.",
        f"Active ({n}): {paint(shown, BOLD, GREEN)}",
        "Downloads will be grouped into dataset/<class>/ subfolders.",
    ]


# ---------------------------------------------------------------------------
# Model Selection Screen Elements
# ---------------------------------------------------------------------------

def model_selection_items(app):
    models = get_available_models()
    items = []
    for m in models:
        exists = (MODELS_DIR / m).exists()
        tag = "✔" if m in app.selected_models else ""
        if not exists:
            tag += " [missing on disk]" if tag else "[missing]"
        items.append(
            Item(
                label=m,
                action=lambda a, name=m: action_toggle_model(a, name),
                detail=tag,
            )
        )
    items.append(Item("Select all models", action_select_all_models))
    items.append(Item("Reset to single model (yolov8n.pt)", lambda a: action_select_single_model(a, "yolov8n.pt")))
    return items


def model_selection_body(app):
    models = get_available_models()
    n_sel = len(app.selected_models)
    lines = [
        "Pick a number to toggle  ✔ on/off. You can select one or multiple models.",
    ]
    if n_sel == 1:
        lines.append(f"Active model: {paint(app.selected_models[0], BOLD, GREEN)} (single-model mode)")
    else:
        shown = ", ".join(app.selected_models)
        lines.append(f"Active models ({n_sel}/{len(models)}): {paint(shown, BOLD, GREEN)} (multi-model mode)")

    dev = app.shared_settings.get("device", "CUDA:0")
    cuda_ok, cuda_desc, gpu_name = get_cuda_status()
    if str(dev).upper().startswith("CUDA") and cuda_ok:
        lines.append(f"Target compute device: {paint(dev, BOLD, CYAN)} [{paint(gpu_name, BOLD, YELLOW)}]")
    else:
        lines.append(f"Target compute device: {paint(dev, BOLD, CYAN)} ({cuda_desc})")
    return lines


# ---------------------------------------------------------------------------
# Screens Definition
# ---------------------------------------------------------------------------

def main_items(app):
    total = _count_all_images()
    n_models = len(app.selected_models)
    dev = app.shared_settings.get("device", "CUDA:0")
    cuda_ok, _, gpu_name = get_cuda_status()
    dev_str = f"{dev} ({gpu_name})" if (str(dev).upper().startswith("CUDA") and cuda_ok and gpu_name) else str(dev)
    n_runs = len(_list_result_runs())
    active_metrics = sum(1 for k, s in METRICS_SCHEMA.items() if s["type"] is bool and app.metrics.get(k, s["default"]))

    model_detail = app.selected_models[0] if n_models == 1 else f"{n_models} selected"

    return [
        Item("Download dataset",            lambda a: a.go("download_dataset")),
        Item("Clear dataset",               lambda a: a.go("clear_dataset"),
             detail=f"{total:,} images" if total else "empty"),
        Item("Select model(s)",             lambda a: a.go("select_models"),
             detail=model_detail),
        Item("Configure shared settings",   lambda a: a.go("configure_shared"),
             detail=dev_str),
        Item("Configure tracked metrics",   lambda a: a.go("configure_metrics"),
             detail=f"{active_metrics} active"),
        Item("Test latency (benchmark)",    lambda a: a.go("test_latency"),
             detail=f"on {dev}"),
        Item("Run tests (eval without training)", lambda a: a.go("run_tests"),
             detail=f"{n_models} model(s)"),
        Item("Run training",                lambda a: a.go("run_training")),
        Item("Clear results",               lambda a: a.go("clear_results"),
             detail=f"{n_runs} run(s)" if n_runs else "empty"),
        Item("All settings",                lambda a: a.go("all_settings")),
    ]


def download_body(app):
    n   = len(app.selected_classes)
    total = app.shared_settings["num_images"]
    lines = []
    if n:
        per   = total // n
        shown = ", ".join(app.selected_classes[:4])
        if n > 4:
            shown += f" +{n - 4} more"
        lines += [
            f"Classes ({n}):  {paint(shown, BOLD, GREEN)}",
            f"~{per:,} images per class  ({total:,} total)",
        ]
        lines += [f"  → dataset/{c}/" for c in app.selected_classes]
    else:
        lines += [
            "Classes:  none — images will be downloaded by id into dataset/",
            f"Images:   {paint(format_value(SHARED_SETTINGS_SCHEMA['num_images'], total), BOLD, GREEN)}",
        ]
    return lines


def download_items(app):
    n = len(app.selected_classes)
    return [
        Item("Choose classes",          lambda a: a.go("select_classes"),
             detail=f"{n} selected" if n else "none (all images)"),
        shared_setting_item(app, "num_images"),
        Item("Start download",          action_start_download),
    ]


def run_training_body(app):
    seeds = app.shared_settings.get("seeds", [app.shared_settings.get("seed", 0)])
    seeds_fmt = ", ".join(str(s) for s in seeds)
    if len(seeds) > 1:
        seeds_fmt += f" ({len(seeds)} seeds)"

    return [
        "Training pipeline configuration:",
        f"  Compute device: {paint(app.shared_settings['device'], BOLD, GREEN)}",
        f"  Active model(s): {paint(', '.join(app.selected_models), BOLD, CYAN)}",
        f"  Random seed(s): {paint(seeds_fmt, BOLD, GREEN)}",
        f"  Epochs:         {paint(app.shared_settings['epochs'], BOLD, GREEN)}",
        f"  Batch size:     {paint(app.shared_settings['batch_size'], BOLD, GREEN)}",
        f"  Learning rate:  {paint(app.shared_settings['learning_rate'], BOLD, GREEN)}",
    ]


def run_tests_body(app):
    n_models = len(app.selected_models)
    dev = app.shared_settings.get("device", "CUDA:0")
    cuda_ok, _, gpu_name = get_cuda_status()
    dev_str = f"{dev} [{gpu_name}]" if (str(dev).upper().startswith("CUDA") and cuda_ok and gpu_name) else str(dev)
    n_images = _count_all_images()
    split = app.shared_settings.get("split", "val")
    conf = app.shared_settings.get("conf", 0.25)
    iou = app.shared_settings.get("iou", 0.70)
    sz = app.shared_settings.get("imgsz", 640)
    half = app.shared_settings.get("half", False)
    plots = app.shared_settings.get("plots", True)
    seeds = app.shared_settings.get("seeds", [app.shared_settings.get("seed", 0)])

    seeds_desc = (
        f"{', '.join(str(s) for s in seeds)} ({len(seeds)} seeds — Mean ± Std evaluation)"
        if len(seeds) > 1
        else str(seeds[0])
    )

    return [
        "Evaluate object detection performance directly on dataset without fine-tuning.",
        f"  • Selected model(s) ({n_models}): {paint(', '.join(app.selected_models), BOLD, CYAN)}",
        f"  • Compute target:           {paint(dev_str, BOLD, GREEN)}",
        f"  • Dataset images:           {paint(f'{n_images:,} images', BOLD, YELLOW)} in dataset/",
        f"  • Global parameters:        imgsz={sz}, conf={conf}, iou={iou}, fp16={'Yes' if half else 'No'}, split={split}",
        f"  • Plots generation:         {'Enabled (confusion matrix, PR curves, batches)' if plots else 'Disabled'}",
        f"  • Random seed(s):           {paint(seeds_desc, BOLD, GREEN)}",
    ]


def run_tests_items(app):
    return [
        Item("Start test evaluation now",   action_run_tests),
        Item("Select models to test",       lambda a: a.go("select_models")),
        Item("Configure global settings",   lambda a: a.go("configure_shared")),
    ]


def all_settings_items(app):
    items = []
    for key in SHARED_SETTINGS_SCHEMA:
        items.append(shared_setting_item(app, key, with_group=True))
    for key, spec in METRICS_SCHEMA.items():
        val = app.metrics.get(key, spec["default"])
        detail = ("✔ ON" if val else "✘ OFF") if spec["type"] is bool else str(val)
        items.append(
            Item(
                label=f"Metric: {spec['label']}",
                action=lambda a, k=key: action_toggle_metric(a, k),
                detail=detail,
            )
        )
    return items


SCREENS = {
    "main": Screen(
        title="Main menu",
        color=MAGENTA,
        body=lambda app: ["Choose an action or configure dataset & model parameters."],
        items=main_items,
    ),
    "download_dataset": Screen(
        title="Download dataset",
        body=download_body,
        items=download_items,
    ),
    "select_classes": Screen(
        title="Select COCO classes",
        body=class_body,
        items=class_items,
    ),
    "clear_dataset": Screen(
        title="Clear dataset",
        color=RED,
        body=lambda app: [
            "Removes downloaded images from dataset/ subfolders.",
            "You will choose which class folder to clear.",
        ],
        items=lambda app: [Item("Clear dataset now", action_clear_dataset)],
    ),
    "select_models": Screen(
        title="Select YOLO models",
        body=model_selection_body,
        items=model_selection_items,
    ),
    "configure_shared": Screen(
        title="Configure shared settings",
        body=lambda app: [
            "Global pipeline, model, and hardware settings.",
            "Model parameters (conf, iou, half, max_det, plots, etc.) apply to all models.",
        ],
        items=lambda app: [shared_setting_item(app, key) for key in SHARED_SETTINGS_SCHEMA],
    ),
    "configure_metrics": Screen(
        title="Configure tracked metrics",
        body=metrics_body,
        items=metrics_items,
    ),
    "clear_results": Screen(
        title="Clear results",
        body=lambda app: ["Removes saved runs and benchmark outputs."],
        items=lambda app: [Item("Clear results now", action_clear_results)],
    ),
    "test_latency": Screen(
        title="Test latency & metrics",
        body=lambda app: [
            f"Active models ({len(app.selected_models)}): {', '.join(app.selected_models)}",
            f"Target device: {app.shared_settings.get('device', 'CUDA:0')}",
            f"Benchmark parameters: {app.metrics.get('iterations', 20)} iterations, {app.metrics.get('warmup_passes', 5)} warmup passes",
            "Measures latency, throughput, speed breakdown, model complexity, and GPU VRAM.",
        ],
        items=lambda app: [
            Item("Run benchmark now", action_test_latency),
            Item("Configure tracked metrics", lambda a: a.go("configure_metrics")),
            Item("Select models to test", lambda a: a.go("select_models")),
            Item("Configure device / shared", lambda a: a.go("configure_shared")),
        ],
    ),
    "run_tests": Screen(
        title="Run tests (evaluation)",
        body=run_tests_body,
        items=run_tests_items,
    ),
    "run_training": Screen(
        title="Run training",
        body=run_training_body,
        items=lambda app: [
            Item("Start training session",   action_run_training),
            Item("Select models",            lambda a: a.go("select_models")),
            Item("Change shared settings",   lambda a: a.go("configure_shared")),
        ],
    ),
    "all_settings": Screen(
        title="All settings",
        body=lambda app: ["Every setting in one place. Choose one to modify it."],
        items=all_settings_items,
    ),
}

# ---------------------------------------------------------------------------
# Navigation Bar & Drawing
# ---------------------------------------------------------------------------

def key_hint(letter):
    return paint(f"[{letter}]", BOLD, YELLOW)


def nav_bar(app):
    def entry(letter, name, target):
        if target is None:
            return paint(f"[{letter}] {name} (none)", DIM)
        return f"{key_hint(letter)} {name} {paint('(' + SCREENS[target].title + ')', GRAY)}"

    back_target    = app.back_stack[-1]    if app.back_stack    else None
    forward_target = app.forward_stack[-1] if app.forward_stack else None
    parts = [
        entry("b", "Back",    back_target),
        entry("f", "Forward", forward_target),
        f"{key_hint('h')} Home",
        f"{key_hint('q')} Quit",
    ]
    return "   ".join(parts)


def draw_screen(app):
    screen = SCREENS[app.current]
    clear_screen()
    banner(screen.title, screen.color)
    print()

    body_lines = screen.body(app)
    if body_lines:
        for line in body_lines:
            print(f"  {line}")
        print()

    items = screen.items(app)
    if items:
        num_w = len(str(len(items)))
        for i, itm in enumerate(items, start=1):
            num = f"{i}.".rjust(num_w + 1)
            det = f"  {paint(itm.detail, GRAY)}" if itm.detail else ""
            print(f"  {paint(num, BOLD, YELLOW)} {itm.label}{det}")
        print()

    if app.message:
        kind, text = app.message
        app.message = None
        say(kind, text)
        print()

    print(nav_bar(app))


def handle_input(app, raw):
    cmd = raw.strip()
    if not cmd:
        return

    lower = cmd.lower()
    if lower == "q":
        app.running = False
        return
    if lower == "h":
        app.go("main")
        return
    if lower == "b":
        app.back()
        return
    if lower == "f":
        app.forward()
        return

    screen = SCREENS[app.current]
    items  = screen.items(app)

    if cmd.isdigit():
        idx = int(cmd) - 1
        if 0 <= idx < len(items):
            items[idx].action(app)
            return
        app.notify("warning", f"Option must be between 1 and {len(items)}.")
        return

    app.notify("warning", f"Unknown command '{cmd}'. Enter a number, or b/f/h/q.")


def run_terminal():
    app = App()
    while app.running:
        draw_screen(app)
        try:
            raw = input(paint("> ", BOLD, MAGENTA))
        except (KeyboardInterrupt, EOFError):
            print()
            break
        handle_input(app, raw)

    print()
    say("info", "Goodbye!")


def main():
    if "--run-tests" in sys.argv:
        app = App()
        action_run_tests(app, auto_confirm=True)
    elif "--clear-results" in sys.argv:
        app = App()
        action_clear_results(app, auto_confirm=True)
    else:
        run_terminal()


if __name__ == "__main__":
    main()