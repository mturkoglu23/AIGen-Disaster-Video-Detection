"""Önerilen modelin mimari şekli (yeniden tasarım) -> results/paper/figs/fig4_architecture.png"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, RESULTS

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix"})
INK, MUTED, LINE = "#1f1f1d", "#5b5a56", "#6d6c68"
BLUE, BLUE_L, BLUE_D = "#2a78d6", "#e3eefb", "#1b4f8f"
ORANGE, ORANGE_L, ORANGE_D = "#eb6834", "#fde8df", "#9c3d15"
AQUA, AQUA_L, AQUA_D = "#1baf7a", "#e2f5ec", "#0e6b4a"
VIOLET, VIOLET_L, VIOLET_D = "#7b5cc4", "#eee9f8", "#4b3590"
PANEL = "#f7f6f3"

# ---- örnek kareler: belirgin hareketli bir video seç
sp = pd.read_csv(os.path.join(RESULTS, "splits.csv"))
best, bv = -1, None
for v in sp[(sp.disaster == "Water")].id.values[:120]:
    x = np.load(os.path.join(CACHE, "clean_pairs", v + ".npy"), mmap_mode="r")
    a, b = x[16].astype(np.float32), x[17].astype(np.float32)
    s = np.abs(b - a).mean() if a.std() > 50 else -1
    if 6 < s < 14 and s > best:
        best, bv = s, v
X = np.load(os.path.join(CACHE, "clean_pairs", bv + ".npy"))
f0, f1 = X[16], X[17]
dmag = np.abs((f1.astype(np.float32) - f0.astype(np.float32)) * 4).mean(-1); dmag = np.clip(dmag / np.percentile(dmag, 99), 0, 1) ** 0.6
others = [X[2 * k] for k in (2, 6, 11)]

fig = plt.figure(figsize=(13, 5.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 130); ax.set_ylim(0, 56); ax.axis("off")


def rbox(x, y, w, h, fc, ec, lw=1.0, r=1.0, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def arrow(p, q, c=LINE, lw=1.1, z=3, ms=10, cs="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=c, zorder=z,
                                 connectionstyle=cs, shrinkA=0, shrinkB=0))


def txt(x, y, s, fs=9, c=INK, **k):
    ax.text(x, y, s, fontsize=fs, color=c, ha=k.pop("ha", "center"), va=k.pop("va", "center"), zorder=k.pop("zorder", 7), **k)


def img(a, x, y, w, h, z=4, cmap=None):
    ax.imshow(a, extent=(x, x + w, y, y + h), zorder=z, cmap=cmap, interpolation="bilinear", aspect="auto")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="white", lw=0.8, zorder=z + 0.05))


def token(x, y, n, c, h=0.85, w=1.5, z=5):
    for i in range(n):
        ax.add_patch(Rectangle((x, y + i * h), w, h, fc=c, ec="white", lw=0.6, zorder=z))


def node(x, y, sym, c, r=1.15):
    ax.add_patch(Circle((x, y), r, fc="white", ec=c, lw=0.95, zorder=5)); txt(x, y + 0.05, sym, fs=10.5, c=c)


# ---- stage panels
for (x0, x1, lab) in [(0.8, 22.5, "(a) Input"), (23.5, 76.5, "(b) Dual-stream encoding (shared over t = 1, …, 16)"),
                      (77.5, 97.6, "(c) Per-frame gated fusion"), (98.6, 129.2, "(d) Temporal aggregation")]:
    rbox(x0, 9.2, x1 - x0, 45.6, PANEL, "#e2e1dc", lw=0.8, r=1.2, z=0)
    txt(x0 + 1.0, 52.9, lab, fs=10.2, ha="left", fontweight="bold", c="#3b3a37")

# ---- (a) input
for k, o in enumerate(others):
    img(o, 2.6 + k * 1.1, 33.2 + k * 1.1, 9.5, 9.5, z=3 + k * 0.1)
img(f0, 2.6 + 3 * 1.1, 33.2 + 3 * 1.1, 9.5, 9.5, z=3.5)
txt(11.6, 31.0, "16 positions, 224×224", fs=8.2, c=MUTED)
img(f0, 2.6, 13.5, 8.6, 8.6); img(f1, 12.6, 13.5, 8.6, 8.6)
txt(6.9, 23.8, "$x_t$", fs=11); txt(16.9, 23.8, "$x_{t+1}$", fs=11)
txt(11.6, 11.3, "frame pair at position $t$", fs=8.2, c=MUTED)

# ---- (b) semantic stream
rbox(24.8, 30.5, 50.6, 20.8, BLUE_L, BLUE, lw=1.0, r=1.0, z=1)
txt(26.0, 49.6, "Semantic stream", fs=9.6, c=BLUE_D, ha="left", fontweight="bold")
txt(74.2, 49.6, "frozen", fs=8.2, c=BLUE_D, ha="right", style="italic")
arrow((15.6, 42.0), (27.0, 41.0), c=BLUE)
bx0, bw, gap = 27.6, 0.82, 0.18
for i in range(24):
    tap = (i + 1) in (8, 16, 24)
    ax.add_patch(Rectangle((bx0 + i * (bw + gap), 34.0), bw, 13.0, fc=BLUE if tap else "#b9d3f2", ec="none", zorder=4))
xend = bx0 + 24 * (bw + gap) - gap
txt((bx0 + xend) / 2, 32.4, "CLIP ViT-L/14 image encoder (24 blocks)", fs=8.2, c=BLUE_D)
cx = 54.2
rbox(cx, 35.3, 5.8, 10.9, "white", BLUE, lw=0.9, r=0.6, z=4)
rows = [("$c^{(8)}$", BLUE), ("$c^{(16)}$", BLUE), ("$c^{(24)}$", BLUE), ("$\\bar{p}$", "#7fb0e8")]
ys = [44.3 - j * 2.5 for j in range(4)]
for (lab, col), yy in zip(rows, ys):
    ax.add_patch(Rectangle((cx + 0.55, yy - 1.05), 1.2, 2.1, fc=col, ec="white", lw=0.5, zorder=5))
    txt(cx + 3.7, yy, lab, fs=8.2, c=BLUE_D)
txt(cx + 2.9, 33.9, "4 × 1024", fs=7.6, c=MUTED)
# taps: block 8 and 16 leave from top, elbow right; block 24 and mean patch leave from right side
for k, yy in [(8, ys[0]), (16, ys[1])]:
    xc = bx0 + (k - 1) * (bw + gap) + bw / 2
    lvl = 48.3 if k == 8 else 47.7
    ax.plot([xc, xc, cx - 0.9], [47.0, lvl, lvl], color=BLUE_D, lw=0.8, zorder=4)
    ax.plot([cx - 0.9, cx - 0.9], [lvl, yy], color=BLUE_D, lw=0.8, zorder=4)
    arrow((cx - 0.9, yy), (cx, yy), c=BLUE_D, lw=0.8, ms=6)
arrow((xend, ys[2]), (cx, ys[2]), c=BLUE_D, lw=0.8, ms=6)
arrow((xend, ys[3]), (cx, ys[3]), c=BLUE_D, lw=0.8, ms=6)
rbox(61.8, 36.6, 7.4, 8.6, "white", BLUE, lw=0.9, r=0.6, z=4)
txt(65.5, 40.9, "LN\nLinear\n4096→256\nGELU", fs=7.6, c=BLUE_D, linespacing=1.15)
arrow((60.0, 40.9), (61.8, 40.9), c=BLUE_D, ms=8)
token(70.9, 37.1, 9, BLUE)
txt(71.65, 46.3, "$s_t$", fs=11.5, c=BLUE_D)
arrow((69.2, 40.9), (70.9, 40.9), c=BLUE_D, ms=8)

# ---- (b) temporal stream
rbox(24.8, 10.6, 50.6, 18.6, ORANGE_L, ORANGE, lw=1.0, r=1.0, z=1)
txt(26.0, 27.5, "Temporal stream", fs=9.6, c=ORANGE_D, ha="left", fontweight="bold")
txt(74.2, 27.5, "trainable", fs=8.2, c=ORANGE_D, ha="right", style="italic")
arrow((21.2, 17.8), (27.0, 17.8), c=ORANGE)
img(f0, 27.4, 16.4, 6.0, 6.0, z=4)
img(dmag, 29.0, 14.8, 6.0, 6.0, z=4.2, cmap="inferno")
txt(31.6, 12.6, "$[\\,x_t;\\;4(x_{t+1}-x_t)\\,]$", fs=8.6, c=ORANGE_D)
txt(31.0, 24.3, "6 channels", fs=7.6, c=MUTED)
xs = 38.4
for i, h in enumerate([10.5, 9.0, 7.4, 5.8, 4.2]):
    x = xs + i * 3.1; yc = 18.4
    ax.add_patch(Polygon([(x, yc - h / 2), (x + 2.0, yc - h / 2 + 0.7), (x + 2.0, yc + h / 2 - 0.7), (x, yc + h / 2)],
                         closed=True, fc=ORANGE if i else "#f39a73", ec="white", lw=0.6, zorder=4))
    if i < 4:
        arrow((x + 2.1, yc), (x + 3.0, yc), c=ORANGE_D, lw=0.7, ms=6)
txt(xs + 7.2, 11.9, "ResNet-18 (6-channel stem)", fs=8.2, c=ORANGE_D)
rbox(55.2, 14.1, 5.6, 8.6, "white", ORANGE, lw=0.9, r=0.6, z=4)
txt(58.0, 18.4, "GAP\n512", fs=7.6, c=ORANGE_D, linespacing=1.15)
arrow((xs + 4 * 3.1 + 2.1, 18.4), (55.2, 18.4), c=ORANGE_D, ms=8)
rbox(61.8, 14.1, 7.4, 8.6, "white", ORANGE, lw=0.9, r=0.6, z=4)
txt(65.5, 18.4, "LN\nLinear\n512→256\nGELU", fs=7.6, c=ORANGE_D, linespacing=1.15)
arrow((60.8, 18.4), (61.8, 18.4), c=ORANGE_D, ms=8)
token(70.9, 14.6, 9, ORANGE)
txt(71.65, 23.8, "$m_t$", fs=11.5, c=ORANGE_D)
arrow((69.2, 18.4), (70.9, 18.4), c=ORANGE_D, ms=8)

# ---- (c) gated fusion
node(80.0, 40.9, "+", BLUE_D); node(80.0, 18.4, "+", ORANGE_D)
arrow((72.4, 40.9), (78.85, 40.9), c=BLUE_D); arrow((72.4, 18.4), (78.85, 18.4), c=ORANGE_D)
txt(80.0, 43.5, "$e_s$", fs=9, c=BLUE_D); txt(80.0, 15.5, "$e_m$", fs=9, c=ORANGE_D)
gx, gy, gw, gh = 82.4, 25.4, 8.2, 8.6
mx = gx + gw / 2
rbox(gx, gy, gw, gh, AQUA_L, AQUA, lw=1.0, r=0.7, z=4)
txt(mx, gy + 6.6, "gate $g$", fs=9, c=AQUA_D, fontweight="bold")
txt(mx, gy + 4.4, "2-layer MLP", fs=7.4, c=MUTED)
ax.add_patch(Rectangle((gx + 1.0, gy + 1.2), (gw - 2.0) * 0.62, 1.4, fc=BLUE, ec="none", zorder=6))
ax.add_patch(Rectangle((gx + 1.0 + (gw - 2.0) * 0.62, gy + 1.2), (gw - 2.0) * 0.38, 1.4, fc=ORANGE, ec="none", zorder=6))
# h -> gate (vertical drop from the + nodes, then into the gate's left side)
ax.plot([80.0, 80.0, gx], [39.75, gy + gh - 2.2, gy + gh - 2.2], color=BLUE_D, lw=0.8, zorder=3)
arrow((gx - 0.6, gy + gh - 2.2), (gx, gy + gh - 2.2), c=BLUE_D, lw=0.8, ms=6)
ax.plot([80.0, 80.0, gx], [19.55, gy + 2.2, gy + 2.2], color=ORANGE_D, lw=0.8, zorder=3)
arrow((gx - 0.6, gy + 2.2), (gx, gy + 2.2), c=ORANGE_D, lw=0.8, ms=6)
txt(83.0, 42.6, "$h_{t,1}$", fs=9.5, c=BLUE_D, ha="left"); txt(83.0, 16.6, "$h_{t,2}$", fs=9.5, c=ORANGE_D, ha="left")
node(mx, 40.9, "×", BLUE_D); node(mx, 18.4, "×", ORANGE_D)
arrow((81.15, 40.9), (mx - 1.15, 40.9), c=BLUE_D); arrow((81.15, 18.4), (mx - 1.15, 18.4), c=ORANGE_D)
arrow((mx, gy + gh), (mx, 39.75), c=AQUA_D, lw=0.9, ms=7); arrow((mx, gy), (mx, 19.55), c=AQUA_D, lw=0.9, ms=7)
txt(mx + 0.5, 36.9, "$w_{t,1}$", fs=8.4, c=BLUE_D, ha="left"); txt(mx + 0.5, 22.2, "$w_{t,2}$", fs=8.4, c=ORANGE_D, ha="left")
sxn = 94.6
node(sxn, 29.7, "+", AQUA_D, r=1.25)
ax.plot([mx + 1.15, sxn], [40.9, 40.9], color=BLUE_D, lw=0.95, zorder=3); arrow((sxn, 40.9), (sxn, 30.95), c=BLUE_D, lw=0.95, ms=7)
ax.plot([mx + 1.15, sxn], [18.4, 18.4], color=ORANGE_D, lw=0.95, zorder=3); arrow((sxn, 18.4), (sxn, 28.45), c=ORANGE_D, lw=0.95, ms=7)
txt(sxn - 1.6, 29.7, "$z_t$", fs=10, c=AQUA_D, ha="right")
txt(87.5, 13.5, "$w_t=\\mathrm{softmax}(g([h_{t,1};h_{t,2}]))$", fs=8.3, c=AQUA_D)
txt(87.5, 11.1, "$z_t=w_{t,1}h_{t,1}+w_{t,2}h_{t,2}$", fs=8.3, c=AQUA_D)

# ---- (d) temporal aggregation
sx = 100.6
arrow((sxn + 1.25, 29.7), (sx - 0.1, 29.7), c=AQUA_D)
token(sx, 25.4, 1, "#8b8a86", h=8.6, w=1.25)
for i in range(6):
    token(sx + 1.6 + i * 1.6, 25.4, 1, AQUA if i % 2 else "#4cb990", h=8.6, w=1.25)
txt(sx + 1.6 + 6 * 1.6 + 0.62, 29.7, "$\\cdots$", fs=10, c=MUTED)
token(sx + 1.6 + 7 * 1.6, 25.4, 1, AQUA, h=8.6, w=1.25)
txt(sx + 0.62, 24.6, "[CLS]", fs=6.8, c=MUTED, rotation=90, va="top")
txt(sx + 9.0, 22.9, "$z_1,\\ldots,z_{16}$ + positional emb.", fs=7.9)
tx, ty, tw, th = 99.6, 37.6, 19.4, 13.8
rbox(tx, ty, tw, th, VIOLET_L, VIOLET, lw=1.0, r=0.8, z=3)
txt(tx + tw / 2, ty + th - 1.4, "Transformer encoder (pre-norm, $d=256$)", fs=7.7, c=VIOLET_D, fontweight="bold")
for lab, yy in [("LN → multi-head self-attention (4 heads)", 46.9), ("LN → FFN 256→512→256", 43.0)]:
    rbox(tx + 0.8, yy - 1.1, tw - 1.6, 2.2, "white", VIOLET, lw=0.6, r=0.35, z=4)
    txt(tx + tw / 2, yy, lab, fs=6.6, c=VIOLET_D)
txt(tx + tw / 2, 39.4, "residual connections, × 2 layers", fs=7.1, c=VIOLET_D, style="italic")
arrow((sx + 6.5, 34.1), (sx + 6.5, ty), c=VIOLET, lw=1.0, ms=9)
ox = 120.2
ax.plot([tx + tw, ox + 4.2], [ty + th / 2, ty + th / 2], color=VIOLET, lw=1.0, zorder=3)
arrow((ox + 4.2, ty + th / 2), (ox + 4.2, 34.0), c=VIOLET, lw=1.0, ms=9)
txt(ox + 4.6, 35.6, "[CLS] out", fs=7.0, c=MUTED, ha="left")
rbox(ox, 25.4, 8.4, 8.6, "white", LINE, lw=0.9, r=0.6, z=4)
txt(ox + 4.2, 29.7, "LN\nDropout 0.3\nLinear", fs=7.6, linespacing=1.15)
rbox(ox + 0.4, 13.0, 7.6, 7.2, "#fff6dc", "#eda100", lw=1.1, r=0.8, z=4)
txt(ox + 4.2, 17.6, "$p(\\mathrm{real})$", fs=11)
txt(ox + 4.2, 14.7, "video-level", fs=7.4, c=MUTED)
arrow((ox + 4.2, 25.4), (ox + 4.2, 20.2))

# ---- training-only strip
rbox(0.8, 0.8, 128.4, 6.6, "white", "#9a9993", lw=0.9, r=1.0, ls=(0, (4, 3)), z=1)
txt(2.4, 4.1, "Training only:", fs=9, ha="left", fontweight="bold", c="#3b3a37")
for s_, x_ in [("auxiliary heads $a_s$, $a_m$ on $\\mathrm{mean}_t\\,h_{t,k}$ (weight $\\lambda=0.3$)", 17.2),
               ("stream dropout ($p=0.2$, never both streams)", 60.0),
               ("BCE with label smoothing (0.025 / 0.975)", 92.0)]:
    ax.add_patch(Circle((x_ - 1.0, 4.1), 0.45, fc="#9a9993", ec="none", zorder=5))
    txt(x_, 4.1, s_, fs=8.4, ha="left")

fig.savefig(os.path.join(RESULTS, "paper", "figs", "fig4_architecture.png"), dpi=300, facecolor="white")
print("ok", bv)
