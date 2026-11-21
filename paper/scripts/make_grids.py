"""Draw the two thumbnail grids (Grid B rows: highest recall, best balance, lowest recall) from the renders in paper/figs/renders/.

The renders are threader outputs (seed 7) of COCO val2017 image 000000579158,
which is not part of the study set. Order follows paper/data/order.txt.
Run from the repo root:  python paper/scripts/make_grids.py
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

P = Path(__file__).resolve().parents[1]
R = P / "figs" / "renders"
F = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSerif-Regular.ttf", 34)
order = (P / "data" / "order.txt").read_text().split()
W, H, pad, lab = 640, 480, 14, 48

def tile(grid, d, im, x, y):
    grid.paste(im.convert("RGB").resize((W, H)), (x, y))
    d.rectangle([x, y, x + W - 1, y + H - 1], outline=(140, 140, 140), width=2)

def centred(d, text, x, y, w):
    d.text((x + (w - d.textlength(text, font=F)) / 2, y), text, fill="black", font=F)

# Grid A: all ten algorithms at 10,000 threads, best to worst
g = Image.new("RGB", (5 * W + 6 * pad, 2 * (H + lab) + 3 * pad), "white"); d = ImageDraw.Draw(g)
for k, m in enumerate(order):
    r, c = divmod(k, 5); x = pad + c * (W + pad); y = pad + r * (H + lab + pad)
    tile(g, d, Image.open(R / f"{m}_10000.png"), x, y)
    centred(d, f"{k + 1}. {m.replace('_', ' ')}", x, y + H + 6, W)
g.save(P / "figs" / "grid_algorithms_10k.png")

# Grid B: best recall, best balance of recall and efficiency, and worst, at five thread counts
cols = [2000, 6000, 10000, 14000, 20000]; top = 54
balance = (P / "data" / "pareto_algo.txt").read_text().split()[0]
rows = [(order[0], "highest recall"), (balance, "best balance of recall and efficiency"), (order[-1], "lowest recall")]
g = Image.new("RGB", (5 * W + 6 * pad, top + len(rows) * (H + lab) + (len(rows) + 1) * pad), "white"); d = ImageDraw.Draw(g)
for c, t in enumerate(cols):
    centred(d, f"{t:,} threads", pad + c * (W + pad), 8, W)
for r, (m, what) in enumerate(rows):
    y = top + pad + r * (H + lab + pad)
    for c, t in enumerate(cols):
        tile(g, d, Image.open(R / f"{m}_{t}.png"), pad + c * (W + pad), y)
    centred(d, f"{m.replace('_', ' ')} ({what})", pad, y + H + 6, 5 * W + 4 * pad)
g.save(P / "figs" / "grid_best_worst.png")
print("order:", order)
