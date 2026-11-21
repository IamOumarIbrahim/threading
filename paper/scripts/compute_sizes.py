"""Measure the file size of every recognition-set image in four forms.

Run from the repo root after re-rendering each image with `threader --npy`:
    python paper/scripts/compute_sizes.py <render_dir> <coco_jpeg_dir>

<render_dir>/<image_id>/ holds the threader output for that image
(one algorithm, its thread count, seed 7). <coco_jpeg_dir> holds the
original COCO val2017 JPEGs.

Sizes written to paper/data/sizes.csv:
  jpeg_bytes    the original COCO JPEG
  png_bytes     the rendered thread image, as stored in IMAGE_DATASET
  packed_bytes  the thread list bit-packed losslessly: 4 coordinates x 10 bits
                + R, G, B, A x 8 bits = 72 bits = 9 bytes per thread
  xz_bytes      the same thread list, stored field by field (all x1, then all
                y1, ...) and compressed with xz (LZMA, preset 9); lossless
  png_match     1 if this re-render is byte-identical to the dataset PNG
"""
import csv, hashlib, lzma, struct, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
render_dir, jpeg_dir = Path(sys.argv[1]), Path(sys.argv[2])
rows = list(csv.DictReader(open(ROOT / "IMAGE_DATASET/manifest.csv")))

out = open(ROOT / "paper/data/sizes.csv", "w", newline="")
w = csv.writer(out)
w.writerow(["image_id", "method", "threads", "n_threads", "jpeg_bytes", "png_bytes",
            "packed_bytes", "xz_bytes", "png_match"])
missing = 0
for r in rows:
    d = render_dir / r["image_id"]
    npys = list(d.glob("*.npy"))
    pngs = list(d.glob("*.png"))
    if not npys:
        missing += 1
        continue
    t = np.load(npys[0])                        # (N, 8) int32: r g b a x1 y1 x2 y2
    n = len(t)
    assert t[:, 4:].max() < 1024 and t[:, 4:].min() >= 0, "coordinate does not fit in 10 bits"
    packed = (n * 72 + 7) // 8
    planar = b"".join(t[:, c].astype(np.uint16 if c >= 4 else np.uint8).tobytes() for c in range(8))
    xz = len(lzma.compress(planar, preset=9 | lzma.PRESET_EXTREME))
    match = int(bool(pngs) and hashlib.sha256(pngs[0].read_bytes()).hexdigest() == r["sha256"])
    w.writerow([r["image_id"], r["algorithm"], r["threads_target"], n,
                (jpeg_dir / r["source"]).stat().st_size, r["bytes"], packed, xz, match])
out.close()
print("missing renders:", missing)
