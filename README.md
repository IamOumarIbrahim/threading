<div align="center">

# Filography: A Revisited Technique for Digital Imagery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Oumar Ibrahim**, **Mohamad Khairy**, **Mir Umer Jarwar** and **Anugrah Shaju**
*University of Sharjah, Sharjah, UAE*
`u22200741@sharjah.ac.ae` | `mishak@sharjah.ac.ae` | `u20105400@sharjah.ac.ae` | `u22102372@sharjah.ac.ae`

</div>

> [!TIP]
> **Work in progress**: the code and the paper are updated as the research progresses.

Filography redraws a photo as thousands of straight coloured threads. This
repository holds the ten threading algorithms, a YOLO detection benchmark on
thread images, a human recognition study, and the paper *Object Recognition in
Thread Art*, which tests a vision-language model (VLM), YOLO and a human rater
on the same thread images.

## Project structure

```
threading/
├── analysis/             # Graph scripts for the benchmark report
├── config/               # Pipeline and recognition study settings
├── docs/                 # Algorithm notes and findings
├── IMAGE_DATASET/        # Study set, round 1: 500 scenes (manifest and labels; images not in git)
├── IMAGE_DATASET_EXTRA/  # Study set, round 2: 500 more scenes (manifest and labels; images not in git)
├── models/               # YOLO nano weights
├── native/               # threader C++ source and Windows build script
├── paper/                # The paper: LaTeX source, figures, data and scripts
├── recognition/          # Human recognition study: dataset builder, web app, scoring
├── reports/              # Benchmark reports and figures
├── src/                  # Python benchmark pipeline
├── tests/                # Automated tests
├── build/                # Compiled threader (generated, not in git)
├── dataset/              # COCO val2017 images and labels (downloaded, not in git)
├── results/              # Benchmark outputs (generated, not in git)
└── runs/                 # YOLO training runs (generated, not in git)
```

## Setup

Python 3.12 or newer.

```bash
pip install -r requirements.txt
python src/download_dataset.py -n 5000
```

Build the threader on Windows with the MSVC C++ build tools:

```bash
native\build.bat
```

or on Linux with GCC:

```bash
g++ -O2 -std=c++17 -I native native/threader.cpp -o build/threader
```

The same seed gives byte-identical images with the same compiler. Four of the
ten algorithms also match across compilers. The other six use C++ random-number
functions that differ between MSVC and GCC, so their threads differ slightly.

## Threading benchmark (YOLO)

This trains one YOLO model per algorithm and thread count, and needs a CUDA GPU:

```bash
python src/run_threaded_experiment.py
```

Add `--daily-pause` to stop GPU work between 15:00 and 23:00 each day. Results go
to `results/`. Copy `threaded_benchmark_results.md` and `.csv` to `reports/`
to publish them, then redraw the figures:

```bash
python analysis/make_graphs.py
python analysis/plot_normalised.py
```

Dataset, model selection and evaluation menus are also available in the terminal:

```bash
python src/run_terminal.py
```

## Paper: Object Recognition in Thread Art

`paper/main.tex` is the IEEE conference paper and `paper/main.pdf` the built copy.
Every number, table and figure comes from the scripts in `paper/scripts/`, so the
paper can be rebuilt from the raw results:

```bash
python paper/scripts/make_data.py        # results, tables and numbers quoted in the text
python paper/scripts/bootstrap.py        # 95% confidence intervals
python paper/scripts/make_efficiency.py  # selection table, heatmaps and Pareto front
python paper/scripts/make_size_only.py   # recall per KB table, drawing time ignored
python paper/scripts/make_grids.py       # thumbnail grids
python paper/scripts/make_internals.py <render_dir>  # appendix: inside view of each algorithm (needs threader --npy on figs/renders/original.jpg)
cd paper && latexmk -pdf main.tex
```

What the study contains:

| Part | Where |
|---|---|
| VLM answers for 1,000 thread images and their 1,000 original photos, plus a repeat run | `paper/vlm/` |
| Prompt and sandboxed script used for every VLM call (one fresh session per image) | `paper/vlm/PROMPT.md`, `paper/vlm/run.sh` |
| Pretrained YOLO on the same images, scored on human COCO labels | `paper/scripts/yolo_pretrained.py`, `paper/data/yolo_pre_*.csv` |
| Drawing time of every algorithm and thread count, measured in C++ | `paper/scripts/time_threader.py`, `paper/scripts/threader_timing.patch`, `paper/data/gen_time.csv` |
| File size of every compressed thread list | `paper/scripts/compute_sizes.py`, `paper/data/sizes*.csv` |
| Selection of the second-round scenes and their blind folders | `paper/scripts/select_extra.py`, `paper/scripts/make_extra_blind.py` |

The images in `IMAGE_DATASET*/images/` are not in git. Rebuild round 1 with
`recognition/build_dataset.py` and round 2 with `paper/scripts/select_extra.py`
and the threader. The COCO annotations (`instances_val2017.json`) come from the
COCO website. The diagram blocks in Fig. 3 use
[PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) (MIT licence),
included in `paper/third_party/`.

## Human recognition study

See [recognition/README.md](recognition/README.md). Rater answers
(`responses.sqlite`) contain personal data and are never committed.

## Tests

```bash
pytest tests
```
