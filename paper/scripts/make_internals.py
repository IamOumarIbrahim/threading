"""Panels for the algorithm diagrams (Figs. on how each algorithm works inside).

Run from the repo root with the project venv, after rendering the scene with
    build/threader paper/figs/renders/original.jpg --threads 10000 --npy --out <dir>
then
    .venv/bin/python paper/scripts/make_internals.py <dir>

For every algorithm this writes three PNG panels to paper/figs/internals/:
<algo>_1.png, <algo>_2.png, <algo>_3.png. Each panel shows an internal state of
the algorithm. Maps that do not depend on random numbers (gradients, the edge
direction field, ink need, the quadtree, the Radon sinogram) are recomputed
exactly from the formulas in native/threader.cpp. Partial canvases use the real
thread order from the .npy file. Panels that show random proposals (candidate
lines, rays) draw fresh samples with the same rules, so they illustrate the
step rather than replay it.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper/figs/renders/original.jpg"
OUT = ROOT / "paper/figs/internals"
OUT.mkdir(exist_ok=True)
NPY = Path(sys.argv[1])
QMIN = 0.08
rng = np.random.default_rng(7)

T = np.asarray(Image.open(SRC).convert("RGB"), dtype=np.float32) / 255.0
H, W, _ = T.shape
GRAY = T.mean(axis=2)


def threads(algo):
    return np.load(NPY / f"original_{algo}.npy")


# ---------------------------------------------------------------- helpers from threader.cpp
def gauss_blur(img, sigma):
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) ** 2) / sigma ** 2)
    k /= k.sum()
    two_d = img.ndim == 2
    a = img[..., None] if two_d else img
    p = np.pad(a, ((0, 0), (r, r), (0, 0)), mode="edge")
    a = sum(k[i] * p[:, i:i + W] for i in range(2 * r + 1))
    p = np.pad(a, ((r, r), (0, 0), (0, 0)), mode="edge")
    a = sum(k[i] * p[i:i + H] for i in range(2 * r + 1))
    return a[..., 0] if two_d else a


def sobel(g):
    p = np.pad(g, 1, mode="edge")
    s = lambda dy, dx: p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
    gx = ((s(-1, 1) + 2 * s(0, 1) + s(1, 1)) - (s(-1, -1) + 2 * s(0, -1) + s(1, -1))) / 8
    gy = ((s(1, -1) + 2 * s(1, 0) + s(1, 1)) - (s(-1, -1) + 2 * s(-1, 0) + s(-1, 1))) / 8
    return gx, gy


def tangent_field():
    tb = gauss_blur(T, 1.0)
    jxx = jxy = jyy = 0
    for c in range(3):
        gx, gy = sobel(tb[..., c])
        jxx, jxy, jyy = jxx + gx * gx, jxy + gx * gy, jyy + gy * gy
    jxx, jxy, jyy = gauss_blur(jxx, 4.0), gauss_blur(jxy, 4.0), gauss_blur(jyy, 4.0)
    bias = 0.5 * (jxx + jyy).mean() + 1e-9
    ph = np.radians(45 + 90)
    nx, ny = np.cos(ph), np.sin(ph)
    phi = 0.5 * np.arctan2(2 * (jxy + bias * nx * ny), (jxx + bias * nx * nx) - (jyy + bias * ny * ny))
    return -np.sin(phi), np.cos(phi), jxx + jyy


def central_gradients():
    g = (np.round(T * 255).sum(axis=2)) / (3 * 255)
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    return gx, gy, np.hypot(gx, gy)


def ink_need():
    return np.clip((1.0 - T) / 1.0, 0, 1).max(axis=2)   # white background


def render(th, bg=1.0):
    prem = np.zeros((H, W, 3), np.float32); alpha = np.zeros((H, W), np.float32)
    for r, g, b, a, x1, y1, x2, y2 in th:
        if a <= 0:
            continue
        n = max(abs(x2 - x1), abs(y2 - y1)) + 1
        t = np.linspace(0, 1, n) if n > 1 else np.zeros(1)
        px = np.floor(x1 + (x2 - x1) * t + 0.5).astype(int)
        py = np.floor(y1 + (y2 - y1) * t + 0.5).astype(int)
        af = a / 255.0
        prem[py, px] = prem[py, px] * (1 - af) + np.array([r, g, b]) / 255.0 * af
        alpha[py, px] = alpha[py, px] * (1 - af) + af
    return np.clip(prem + bg * (1 - alpha[..., None]), 0, 1)


# ---------------------------------------------------------------- drawing
def panel(name, base=None, cmap=None, vmin=None, vmax=None, lines=(), points=(), crop=None):
    """Save a 640x480 panel: a base image (RGB or scalar map) with optional overlays.
    lines: list of (segments Nx2x2, colour, width, alpha); points: (xy, colour, size)."""
    fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    if base is not None:
        ax.imshow(base, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest" if crop else "antialiased")
    for seg, col, lw, al in lines:
        ax.add_collection(LineCollection(seg, colors=col, linewidths=lw, alpha=al, capstyle="round"))
    for xy, col, s in points:
        ax.scatter(xy[:, 0], xy[:, 1], s=s, c=col, linewidths=0)
    x0, y0, x1, y1 = crop if crop else (0, 0, W, H)
    ax.set_xlim(x0 - 0.5, x1 - 0.5); ax.set_ylim(y1 - 0.5, y0 - 0.5)
    fig.savefig(OUT / f"{name}.png", dpi=100)
    plt.close(fig)


def segs(th):
    return th[:, [4, 5, 6, 7]].reshape(-1, 2, 2).astype(float)


def faded(img, k=0.35):
    return 1 - k * (1 - img)


def norm(a, p=99):
    return np.clip(a / (np.percentile(a, p) + 1e-9), 0, 1)


CROP = (300, 150, 460, 270)   # cockpit and engine of the aircraft, 4:3

# ---------------------------------------------------------------- shared maps
VX, VY, TENSOR = tangent_field()
GX, GY, MAG = central_gradients()
Q = ink_need()


def field_lines(step=12, half=5):
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    vx, vy = VX[ys, xs], VY[ys, xs]
    return np.stack([np.stack([xs - vx * half, ys - vy * half], -1),
                     np.stack([xs + vx * half, ys + vy * half], -1)], -2).reshape(-1, 2, 2)


# 1. thread painter ------------------------------------------------------------
th = threads("thread_painter")
k = 0
canvas = np.ones_like(T)
err = ((T - canvas) ** 2).sum(axis=2)
panel("thread_painter_1", err.clip(0, np.percentile(err, 99.5)), cmap="magma")
# candidates for thread k+1: centres drawn from the error map, angle along the edge field (60%)
p = (err / err.sum()).ravel()
idx = rng.choice(W * H, size=160, p=p)
cy, cx = idx // W, idx % W
ang = np.where(rng.random(160) < 0.6, np.arctan2(VY[cy, cx], VX[cy, cx]) + rng.normal(0, 0.12, 160), rng.random(160) * np.pi)
L = np.exp(np.log(0.3 * np.hypot(W, H)) + rng.normal(0, 0.6, 160)) / 2
cand = np.stack([np.stack([cx - np.cos(ang) * L, cy - np.sin(ang) * L], -1),
                 np.stack([cx + np.cos(ang) * L, cy + np.sin(ang) * L], -1)], 1)
panel("thread_painter_2", faded(T, 0.3), lines=[(cand, "#4477aa", 0.9, 0.7), (segs(th[k:k + 1]), "#cc3311", 3.5, 1.0)])
panel("thread_painter_3", render(th[:2000]))

# 2. greedy residual -----------------------------------------------------------
th = threads("greedy_residual")
k = 1500
canvas = render(th[:k])
err = ((T - canvas) ** 2).sum(axis=2)
panel("greedy_residual_1", err.clip(0, np.percentile(err, 99.5)), cmap="magma")
p = (err / err.sum()).ravel()
idx = rng.choice(W * H, size=20, p=p)
cy, cx = idx // W, idx % W
ang = np.where(rng.random(20) < 0.6, np.arctan2(VY[cy, cx], VX[cy, cx]) + rng.normal(0, 0.3, 20), rng.random(20) * np.pi)
Lmax = min(W, H) * 0.5625 * 0.12 ** (k / len(th)); Lmin = max(8, 0.2 * Lmax)
L = (Lmin + rng.random(20) * (Lmax - Lmin)) / 2
cand = np.stack([np.stack([cx - np.cos(ang) * L, cy - np.sin(ang) * L], -1),
                 np.stack([cx + np.cos(ang) * L, cy + np.sin(ang) * L], -1)], 1)
panel("greedy_residual_2", faded(canvas, 0.35), lines=[(cand, "#4477aa", 1.6, 0.9), (segs(th[k:k + 1]), "#cc3311", 3.0, 1.0)])
panel("greedy_residual_3", render(th[:5000]))

# 3. quadtree hatch ------------------------------------------------------------
th = threads("quadtree_hatch")
img = np.round(T * 255)


def box_priority(x0, y0, x1, y1):
    area = (x1 - x0) * (y1 - y0)
    if area <= 16 or x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    c = img[y0:y1, x0:x1].reshape(-1, 3)
    return -(np.sqrt(c.var(axis=0)).mean() * area ** 0.65)


import heapq
target = max(64, min(len(th) // 3, 2500))
uid = 0
pq = [(box_priority(0, 0, W, H), 0, 0, 0, W, H)]
while len(pq) < target:
    pr, u, x0, y0, x1, y1 = heapq.heappop(pq)
    if pr == 0.0 or x1 - x0 < 6 or y1 - y0 < 6:
        uid += 1; heapq.heappush(pq, (0.0, uid, x0, y0, x1, y1)); break
    mx, my = (x0 + x1) // 2, (y0 + y1) // 2
    for b in ((x0, y0, mx, my), (mx, y0, x1, my), (x0, my, mx, y1), (mx, my, x1, y1)):
        uid += 1; heapq.heappush(pq, (box_priority(*b), uid, *b))
leaves = [b[2:] for b in pq]
rects = []
for x0, y0, x1, y1 in leaves:
    rects += [[(x0, y0), (x1, y0)], [(x1, y0), (x1, y1)], [(x1, y1), (x0, y1)], [(x0, y1), (x0, y0)]]
panel("quadtree_hatch_1", faded(T, 0.6), lines=[(np.array(rects, float) - 0.5, "#cc3311", 0.5, 0.9)])
# threads per leaf, as a map
wts = []
for x0, y0, x1, y1 in leaves:
    c = img[y0:y1, x0:x1].reshape(-1, 3); n = len(c)
    wts.append((np.sqrt(((c - c.mean(0)) ** 2).sum() / (3 * n)) + 5) * np.sqrt(n))
wts = np.array(wts) / sum(wts) * len(th)
dens = np.zeros((H, W))
for (x0, y0, x1, y1), w in zip(leaves, wts):
    dens[y0:y1, x0:x1] = w / ((x1 - x0) * (y1 - y0))
panel("quadtree_hatch_2", np.sqrt(dens), cmap="viridis")
panel("quadtree_hatch_3", render(th), crop=CROP)

# 4. radon peel ---------------------------------------------------------------
th = threads("radon_peel")
A = 120
c0x, c0y = (W - 1) / 2, (H - 1) / 2
R = int(np.ceil(np.hypot(c0x, c0y))) + 2
E = np.where(Q > QMIN, Q, 0)
ys, xs = np.nonzero(E)
th_a = np.arange(A) * np.pi / A
S = np.zeros((A, 2 * R + 1))
for a in range(A):
    b = np.floor((xs - c0x) * np.cos(th_a[a]) + (ys - c0y) * np.sin(th_a[a]) + 0.5).astype(int) + R
    np.add.at(S[a], b, E[ys, xs])
panel("radon_peel_1", Q, cmap="gray_r", vmin=0, vmax=1)
a_best, b_best = np.unravel_index(S.argmax(), S.shape)
fig = plt.figure(figsize=(6.4, 4.8), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
ax.imshow(S.T, cmap="inferno", aspect="auto", origin="lower")
ax.scatter([a_best], [b_best], s=260, facecolors="none", edgecolors="#33bbee", linewidths=2.5)
fig.savefig(OUT / "radon_peel_2.png", dpi=100); plt.close(fig)
panel("radon_peel_3", faded(render(th[:3000]), 1.0), lines=[(segs(th[:40]), "#cc3311", 1.8, 1.0)])

# 5. edge tangent --------------------------------------------------------------
th = threads("edge_tangent")
panel("edge_tangent_1", norm(MAG), cmap="gray_r")
cx0, cy0, cx1, cy1 = CROP
ys, xs = np.mgrid[cy0 + 2:cy1:5, cx0 + 2:cx1:5]
gx, gy, m = GX[ys, xs], GY[ys, xs], norm(MAG)[ys, xs]
keep = m > 0.12
tang = np.arctan2(gy, gx) + np.pi / 2
h = 2.4
seg = np.stack([np.stack([xs - np.cos(tang) * h, ys - np.sin(tang) * h], -1),
                np.stack([xs + np.cos(tang) * h, ys + np.sin(tang) * h], -1)], -2)[keep]
panel("edge_tangent_2", faded(T, 0.5), lines=[(seg, "#cc3311", 2.0, 1.0)], crop=CROP)
panel("edge_tangent_3", render(th[:3000]))

# 6. flow streamlines ----------------------------------------------------------
th = threads("flow_streamlines")
panel("flow_streamlines_1", faded(np.repeat(GRAY[..., None], 3, 2), 0.35),
      lines=[(field_lines(14, 6), "#004488", 1.6, 1.0)])
Qd = np.where(Q > QMIN, Q, 0)
C = 4
deficit = Qd[:H // C * C, :W // C * C].reshape(H // C, C, W // C, C).sum(axis=(1, 3))
panel("flow_streamlines_2", np.kron(deficit, np.ones((C, C))), cmap="viridis")


def trace(x0, y0):
    halves = []
    for sign in (1, -1):
        x, y, pts, pd = x0, y0, [], None
        for _ in range(70):
            ix, iy = int(x + 0.5), int(y + 0.5)
            if not (0 <= ix < W and 0 <= iy < H):
                break
            dx, dy = VX[iy, ix], VY[iy, ix]
            if pd is None:
                dx, dy = dx * sign, dy * sign
            else:
                if dx * pd[0] + dy * pd[1] < 0:
                    dx, dy = -dx, -dy
                if dx * pd[0] + dy * pd[1] < 0.5:
                    break
            nx_, ny_ = x + 3 * dx, y + 3 * dy
            if not (0 <= nx_ <= W - 1 and 0 <= ny_ <= H - 1) or deficit[min(int(ny_) // C, H // C - 1), min(int(nx_) // C, W // C - 1)] < 1:
                break
            pts.append((nx_, ny_)); x, y, pd = nx_, ny_, (dx, dy)
        halves.append(pts)
    return np.array(halves[1][::-1] + [(x0, y0)] + halves[0])


p = np.clip(deficit - 2, 0, None).ravel(); p /= p.sum()
stream, dots = [], []
for idx in rng.choice(len(p), 14, p=p):
    pts = trace((idx % (W // C) + 0.5) * C, (idx // (W // C) + 0.5) * C)
    if len(pts) > 3:
        stream.append(pts[:, None, :])
        cut = pts[::6]
        dots.append(cut)
        stream_chords = np.stack([cut[:-1], cut[1:]], 1)
        stream.append(stream_chords)
panel("flow_streamlines_3", faded(T, 0.55),
      lines=[(np.concatenate([s for s in stream if s.shape[1] == 2]), "#cc3311", 2.2, 1.0)],
      points=[(np.concatenate(dots), "#0077bb", 14)])

# 7. greedy error --------------------------------------------------------------
th = threads("greedy_error")
panel("greedy_error_1", norm(MAG * 2.5 + 0.02), cmap="gray_r")
rgba = np.round(T * 255)
max_len = max(5, int(min(W, H) * 0.06))
rays, best = [], []
cx0, cy0, cx1, cy1 = CROP
for _ in range(12):
    cand = [(rng.integers(cy0 + 10, cy1 - 10), rng.integers(cx0 + 10, cx1 - 10)) for _ in range(24)]
    y1, x1 = max(cand, key=lambda c: MAG[c])
    errs = []
    for d in range(8):
        t = d * 2 * np.pi / 8; ln = rng.integers(3, max_len + 1)
        x2 = int(np.clip(round(x1 + np.cos(t) * ln), 0, W - 1)); y2 = int(np.clip(round(y1 + np.sin(t) * ln), 0, H - 1))
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        e = np.abs(rgba[y2, x2] - rgba[y1, x1]).sum() + np.abs(rgba[my, mx] - rgba[y1, x1]).sum()
        errs.append((e, x2, y2)); rays.append([(x1, y1), (x2, y2)])
    e, x2, y2 = min(errs); best.append([(x1, y1), (x2, y2)])
panel("greedy_error_2", faded(T, 0.5), crop=CROP, lines=[(np.array(rays, float), "#4477aa", 1.4, 0.9), (np.array(best, float), "#cc3311", 3.0, 1.0)])
panel("greedy_error_3", render(th[:3000]))

# 8. scanline dashes ----------------------------------------------------------
th = threads("scanline_dashes")
Qs = np.where(Q > QMIN, Q, 0)
M = max(3, round(Qs.sum() / (len(th) * 1.08)))
panel("scanline_dashes_1", Qs, cmap="gray_r", vmin=0, vmax=1)
row = 230
cum = np.cumsum(Qs[row])
fig = plt.figure(figsize=(6.4, 4.8), dpi=100); ax = fig.add_axes([0.13, 0.13, 0.84, 0.83])
ax.plot(np.arange(W), cum, color="black", lw=1.6)
for kk in range(1, int(cum[-1] // M) + 1):
    ax.axhline(kk * M, color="#bbbbbb", lw=0.5)
    xk = np.searchsorted(cum, kk * M); ax.plot([xk, xk], [0, kk * M], color="#cc3311", lw=0.6)
ax.set_xlim(0, W); ax.set_ylim(0, cum[-1] * 1.03)
ax.set_xlabel("x along one row", fontsize=16); ax.set_ylabel("ink so far", fontsize=16)
ax.tick_params(labelsize=12)
fig.savefig(OUT / "scanline_dashes_2.png", dpi=100); plt.close(fig)
panel("scanline_dashes_3", render(th), crop=CROP)

# 9. anchor web ---------------------------------------------------------------
th = threads("anchor_web")
n_anchors = max(300, int(len(th) * 0.45))
w = (MAG + 4.0).ravel(); w /= w.sum()
anc = rng.choice(W * H, size=n_anchors, replace=False, p=w)
ax_, ay_ = anc % W, anc // W
panel("anchor_web_1", faded(T, 0.5), points=[(np.stack([ax_, ay_], 1), "#cc3311", 3.0)])
strip_w = max(8, W // 25)
order = np.argsort((ax_ // strip_w) * H + ay_, kind="stable")
path = np.stack([ax_[order], ay_[order]], 1)[:1200]
strips = [[(x, 0), (x, H)] for x in range(strip_w, W, strip_w)]
panel("anchor_web_2", faded(T, 0.35), lines=[(np.array(strips, float), "#999999", 0.6, 1.0),
                                             (np.stack([path[:-1], path[1:]], 1).astype(float), "#0077bb", 0.7, 0.9)])
panel("anchor_web_3", render(th), crop=CROP)

# 10. stipple tour ------------------------------------------------------------
th = threads("stipple_tour")
p99 = max(np.percentile(Q, 99), 1e-3)
qb = gauss_blur(np.clip(Q / p99, 0, 1), 1.5)
panel("stipple_tour_1", qb, cmap="gray_r", vmin=0, vmax=1)
pts = th[:, [4, 5]].astype(float)
panel("stipple_tour_2", None, points=[(pts, "black", 1.2)])
panel("stipple_tour_3", render(th), crop=CROP)

# input and outputs at panel size, so every thumbnail has the same shape
Image.open(SRC).convert("RGB").save(OUT / "input.png")
for algo in ("thread_painter", "greedy_residual", "quadtree_hatch", "radon_peel", "edge_tangent",
             "flow_streamlines", "greedy_error", "scanline_dashes", "anchor_web", "stipple_tour"):
    Image.open(NPY / f"original_{algo}.png").convert("RGB").save(OUT / f"{algo}_out.png")
print("panels written to", OUT)

# keep the PDF small: store every panel as a JPEG
for png in OUT.glob("*.png"):
    Image.open(png).convert("RGB").resize((480, 360), Image.LANCZOS).save(png.with_suffix(".jpg"), quality=85)
    png.unlink()
