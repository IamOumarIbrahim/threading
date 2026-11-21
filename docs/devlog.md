# Devlog

Every decision that changes the study, the paper or the data pipeline gets an
entry here, newest first. Each entry says what changed, why, and which files it
touched. See `AGENTS.md` for the rule.

## Frozen layers

Changing any of these needs an explicit author decision, recorded as a new entry.

| Layer | Frozen value | Frozen on |
|---|---|---|
| Algorithm | thread painter; quadtree hatch only when drawing time matters | 2026-11-22 |
| Size measure | compressed thread list (planar fields, xz preset 9 extreme), compared with the original JPEG | 2026-11-22 |
| Scoring | COCO classes as ground truth, recall, the synonym list in `recognition/score.py` and `recognition/vocabulary.py`, YOLO confidence 0.3 | 2026-11-22 |
| Data | the 1,000 VLM-set thread images (`IMAGE_DATASET`, `IMAGE_DATASET_EXTRA`), the 100 balanced scenes (`paper/scripts/balanced_prepare.py`), render seed 7 | 2026-11-22 |
| Human protocol | the rating app as deployed: 3 raters, all 1,000 images, own spaced random order | 2026-11-22 |

Open (not frozen): thread count (4,000 to 6,000, settle after the balanced run and
human results); machine judge for later papers (pretrained YOLO if the balanced
ranking matches the VLM); human agreement.

## 2026-11-22

### Balanced check done
- 100 scenes (one per cell of the VLM set), each resized to 0.31 megapixels, drawn
  by all 10 algorithms at 2,000 to 20,000 threads (10,000 images), scored by
  pretrained YOLOv12n. Ranking matches main-set YOLO at Spearman 0.94 and the VLM
  at 0.90: the scene assignment does not drive the main results.
  Paper: Section IV-E paragraph, Table IV, Limitations.
  Data: `paper/data/balanced.csv`, `balanced.tex`, `tab_balanced.tex`.
- **Open point for the authors:** YOLO ranks greedy residual first (33.4%
  recall) and thread painter second (27.9%), as it already did on the main set;
  the VLM ranks thread painter first. Greedy residual draws about 16 times faster
  (median 1.2 s vs 19.0 s per image). The frozen algorithm (thread painter) is
  unchanged; revisit only by explicit decision.

### Study folders moved into the repo
- The four VLM blind-set folders (`blind_set`, `blind_set_original`,
  `blind_set_extra`, `blind_set_extra_original`, about 760 MB) moved from
  `~/Desktop/Dev/` to `study/`, which is git-ignored. `make_extra_blind.py` now
  points there.
- `HUMAN_STUDY_1000` (live rating server data) not moved yet: it needs the server
  stopped and restarted with the new `data_dir`.

### Shorter captions
- Every figure and table caption cut to at most 22 words (longest was 109).
  Reading guides moved into the body text: heatmap layout and log scale (Section
  IV-B), efficiency units and table notation (Section IV-E), best/worst algorithm
  names for the size plot (Section IV-D), appendix panel details (Appendix).
  Script-written captions changed in `make_data.py`, `make_efficiency.py`,
  `make_size_only.py`, `balanced_yolo.py`.

### One grid, fine-tuned YOLO removed
- **Grid:** every judge uses 2,000 to 20,000 threads in steps of 2,000, the grid
  the human raters already see. A split grid and trimming the website to 10,000
  were discussed but never applied; the website and all ratings are unchanged.
- **Fine-tuned YOLO removed** from the paper. It used another grid (1,000 to
  10,000 in steps of 1,000), its own boxes as ground truth, mAP50 and training on
  thread images, and its weights are gone. Retraining (about 500,000 renders and
  100 models) is out of scope. Replaces the earlier entry that kept it for retention.
- **YOLO is now pretrained YOLOv12n everywhere**, scored by recall against COCO
  classes on the same 1,000 images as the VLM; retention = recall on thread images
  / recall on originals. `paper/scripts/make_data.py` rewritten for this.
  VLM-YOLO rank correlation rises from 0.88 to 0.98. "Algorithm matters about twice
  as much as five times more threads" still holds (VLM 67 vs 28 points, YOLO 26 vs 12).
- For reference, on 2,000 to 10,000 threads the old fine-tuned YOLO kept 21.2%
  on average against 4.4% for pretrained (not directly comparable: other metric
  and ground truth). Future Work mentions the pilot without the number; the
  native-detector paper will use fine-tuned YOLO as its baseline.
- Algorithm order (by VLM retention over all counts) changed: greedy error now 5th,
  edge tangent 7th. Fig. 2 grid regenerated, appendix rows reordered.
- Balanced check extended to 2,000 to 20,000 threads.

### Freeze decision
- Authors froze algorithm, size measure, scoring and data (table above). Human
  protocol was already frozen.

### Robustness of the choice (done)
- `paper/scripts/robustness.py` recomputes the algorithm ranking and choice with
  recall per scene (instead of pooled over classes), with only 2,000 to 10,000
  threads (15 scenes in every cell), and with both. Thread painter stays the most
  recall per KB for both models in every variant; rankings correlate with the main
  ones at Spearman >= 0.88. The best thread count moves between 4,000 and 6,000,
  so the carried thread count is now stated as a range. Paper: Section IV-E and
  Conclusion.
- 37 of 1,000 thread images have fewer than 90% of their target threads (scanline
  dashes, flow streamlines, radon peel). Stated in Method; file sizes already use
  the threads actually drawn.
- Method now introduces pretrained YOLO explicitly as the zero-shot YOLO judge used
  for heatmaps and selection; fine-tuned YOLO stays for retention only.
- Limitations: fine-tuned YOLO cannot be re-scored (weights not kept); VLM-set
  scenes vary from 0.04 to 0.41 megapixels.

### Standardisation pass (in progress)
- **Balanced check with pretrained YOLO.** The VLM study gives each scene to one
  cell only, so scene difficulty is mixed with the algorithm. Re-running the VLM
  is too costly, so a balanced check uses pretrained YOLOv12n instead: 100 scenes
  (one from each cell of the VLM set), every algorithm, 2,000 to 10,000 threads.
  Scripts: `paper/scripts/balanced_prepare.py`, `paper/scripts/balanced_render.sh`.
- **Fixed exposure.** In the balanced check every scene is resized to the same
  area, 640 x 480 = 307,200 pixels, keeping its aspect ratio, so a thread count
  means the same density everywhere. The main VLM set keeps native sizes
  (38,720 to 409,600 pixels); this is stated as a limitation.
- **Drawing time** is re-measured on the 100 balanced scenes, from the threader log,
  instead of on one 640 x 480 photo.
- **Fine-tuned YOLO is not retrained.** The weights are gone, and retraining means
  100 models on about 500,000 renders. Pretrained YOLO becomes the main YOLO judge,
  because it has the same zero-shot exposure as the VLM and the raters. The
  fine-tuned YOLO ground truth (its own boxes) stays in Limitations.
- **Human protocol frozen as is.** No changes to the rating app mid-study.

### Human raters
- Three raters, one of them the first author, each rate all 1,000 VLM-set thread
  images (confirmed from the live admin page). Paper text updated from "one rater,
  500 images". Own random order per rater, neighbours never share an algorithm or
  thread count. Files: `paper/main.tex`, `paper/figs/models.tex`.

### Algorithm diagrams
- Appendix Figs. 10 and 11 show three internal states per algorithm, recomputed
  from the formulas in `native/threader.cpp`. Placed in an appendix because two
  full-page floats in Method pushed every later figure to the end.
  Files: `paper/scripts/make_internals.py`, `paper/figs/internals.tex`.

### Carry-forward setting
- Paper 1 is framed as the benchmark that fixes the layers for later papers.
- Rule: algorithm with the most recall per KB (drawing time ignored) is thread
  painter; thread count with the most recall per KB for both models together is
  6,000 (about 45 KB). Computed in `paper/scripts/make_size_only.py`.
- Quadtree hatch stays the best balance when drawing time counts.

### LoRa pilot
- Thread painter at 100 to 1,000 threads on 50 scenes: 1 to 2 KB only at 100 to
  200 threads, where pretrained YOLO finds nothing. LoRa needs a more compact
  coding per thread, not fewer threads. Future Work updated.

## 2026-11-21

- Title changed to "Object Recognition in Thread Art by Humans, Vision-Language
  Models and Object Detectors" (no colon titles).
- Affiliation line added.
- Old Fig. 10 (Pareto plot) removed; references rewired to Section IV-E.
- Fig. 3 label overlaps fixed.
- Series of conference papers, not a journal: Future Work lists the detector,
  region-aware privacy, LoRa feasibility and the generator.
