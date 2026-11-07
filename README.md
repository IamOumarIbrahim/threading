<div align="center">

# Filography: A Revisited Technique for Digital Imagery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


**Oumar Ibrahim** and **Mohamad Khairy**  
*Department of Computer Engineering, University of Sharjah, Sharjah, UAE*  
`u22200741@sharjah.ac.ae` | `mishak@sharjah.ac.ae`

</div>


> [!TIP]
> **Work in progress**: The code will be updated as the research progresses.


## Project Structure

```
filography/
├── analysis/          # Graph scripts for the benchmark report
├── config/            # Pipeline and recognition study settings
├── dataset/           # COCO val2017 images and labels (downloaded, not in git)
├── docs/              # Algorithm notes and findings
├── models/            # YOLO nano weights
├── native/            # threader C++ source and build script
├── recognition/       # Human recognition study: dataset builder, web app, scoring
├── reports/           # Benchmark reports and figures
├── src/               # Python benchmark pipeline
├── tests/             # Automated tests
├── build/             # Compiled threader (generated, not in git)
├── results/           # Experiment outputs (generated, not in git)
└── runs/              # YOLO training runs (generated, not in git)
```

## Setup

Python 3.12 and Windows with the MSVC C++ build tools.

```bash
pip install -r requirements.txt
native\build.bat
python src/download_dataset.py -n 5000
```

`native\build.bat` compiles `build\threader.exe`.

## Reproduce

Run the threading benchmark. This trains one YOLO model per algorithm and thread
count, and needs a CUDA GPU:

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

Threader output is deterministic: the same seed gives byte-identical PNGs.

## Interactive terminal

Dataset, model selection and evaluation menus:

```bash
python src/run_terminal.py
```

## Human recognition study

See [recognition/README.md](recognition/README.md).

## Tests

```bash
pytest tests
```
# threading
