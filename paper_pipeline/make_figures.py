"""Makale şekilleri -> results/paper/figs/*.png (300 dpi)."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, RESULTS

FIG = os.path.join(RESULTS, "paper", "figs"); os.makedirs(FIG, exist_ok=True)
P = os.path.join(RESULTS, "paper")
EX = os.path.join(RESULTS, "explain", "P_clip_semtmp_s1")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True})
BLUE, ORANGE, AQUA, YELLOW, GREY, DGREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b9b8b3", "#52514e"
sp = pd.read_csv(os.path.join(RESULTS, "splits.csv"))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)


def frame(vid, data="clean_pairs", t=8):
    x = np.load(os.path.join(CACHE, data, vid + ".npy"), mmap_mode="r")
    return np.array(x[2 * t] if data == "clean_pairs" else x[t])


# ------------------------------------------------------------- Fig. 2: veri örnekleri
dis = sorted(sp.disaster.unique())
rng = np.random.default_rng(3)
fig, ax = plt.subplots(2, 7, figsize=(10, 3.3))
for j, d in enumerate(dis):
    for i, lab in enumerate(["real", "fake"]):
        c = sp[(sp.disaster == d) & (sp.label == lab)].id.values
        v = rng.choice(c)
        while frame(v).std() < 45:
            v = rng.choice(c)
        ax[i, j].imshow(frame(v)); ax[i, j].set_xticks([]); ax[i, j].set_yticks([]); ax[i, j].grid(False)
        for s in ax[i, j].spines.values():
            s.set_visible(False)
        if i == 0:
            ax[i, j].set_title({"Water": "Flood", "Volcanic": "Volcanic eruption"}.get(d, d), fontsize=9)
        if j == 0:
            ax[i, j].set_ylabel(lab.capitalize(), fontsize=10)
fig.tight_layout(pad=0.3); save(fig, "fig2_samples.png")

# ------------------------------------------------------------- Fig. 3: ön işleme (ham vs temiz)
def letterboxed(v):
    r = np.load(os.path.join(CACHE, "raw_hybrid", v + ".npy"), mmap_mode="r")[4]
    return r[:20].mean() < 20 or r[-20:].mean() < 20
cand_real = [v for v in sp[sp.label == "real"].id.values[::7] if letterboxed(v)][:2]
cand_fake = sp[sp.label == "fake"].id.values[[5, 300]].tolist()
sel = cand_fake[:2] + cand_real[:2]
fig, ax = plt.subplots(2, 4, figsize=(8.6, 4.6))
for j, v in enumerate(sel):
    ax[0, j].imshow(frame(v, "raw_hybrid", 4)); ax[1, j].imshow(frame(v, "clean_pairs", 4))
    ax[0, j].set_title(("Fake" if j < 2 else "Real") + f" #{j % 2 + 1}", fontsize=9)
    for i in range(2):
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([]); ax[i, j].grid(False)
ax[0, 0].set_ylabel("Full frame resized\n(aspect ratio distorted)", fontsize=8.5)
ax[1, 0].set_ylabel("Shortcut-controlled\n(bands + 10% margin removed)", fontsize=8.5)
fig.tight_layout(pad=0.4); save(fig, "fig3_preprocessing.png")
print("ön işleme örnekleri:", sel)

# ------------------------------------------------------------- Fig. 4: mimari
fig, ax = plt.subplots(figsize=(10, 4.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")


def box(x, y, w, h, txt, fc, ec=DGREY, fs=8.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arr(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=DGREY))


box(0.5, 17, 11, 10, "Input video\n16 positions\n× (t, t+1)\n224×224", "#f3f2ee")
box(15, 30, 17, 9, "Frame $x_t$", "#f3f2ee")
box(15, 5, 17, 9, "$[\\,x_t,\\ 4(x_{t+1}-x_t)\\,]$\n6-channel input", "#f3f2ee")
arr(11.5, 23.5, 15, 34.5); arr(11.5, 20.5, 15, 9.5)
box(36, 28, 19, 13, "Frozen CLIP ViT-L/14\nCLS of layers 7, 15, 23\n+ mean patch token\n→ LN → Linear(4096→256)", "#dce9f9", ec=BLUE)
box(36, 3, 19, 13, "Temporal stream\nResNet-18 (6-ch, trainable)\n→ LN → Linear(512→256)", "#fde3d8", ec=ORANGE)
arr(32, 34.5, 36, 34.5); arr(32, 9.5, 36, 9.5)
box(59, 15, 12, 14, "Per-frame\ngated fusion", "#e9f6ef", ec=AQUA, fs=8.2)
ax.text(65, 12.4, "$w_t=\\mathrm{softmax}(g([s_t;m_t]))$\n$z_t=w_{t,1}s_t+w_{t,2}m_t$", ha="center", va="top", fontsize=7.6)
arr(55, 34.5, 62, 29); arr(55, 9.5, 62, 15)
ax.text(56.5, 36, "$s_t$", fontsize=9); ax.text(56.5, 6.5, "$m_t$", fontsize=9)
box(75, 15, 12, 14, "[CLS] + 16 tokens\n+ positional emb.\n2-layer Transformer\nencoder", "#f3f2ee", fs=7.8)
arr(71, 22, 75, 22)
box(90.5, 17, 9, 10, "Linear\n→ p(real)", "#f3f2ee")
arr(87, 22, 90.5, 22)
ax.text(45.5, 21.8, "auxiliary heads (training only)\nstream dropout p = 0.2", ha="center", fontsize=7.4, color=DGREY, style="italic")
save(fig, "fig4_architecture.png")

# ------------------------------------------------------------- Fig. 5: sağlamlık
T2 = pd.read_csv(os.path.join(P, "t_robust.csv"))
names = {"ResNet50-SE-BiLSTM-MHA": "ResNet50-SE-\nBiLSTM-MHA", "Xception (frame-level)": "Xception", "EfficientNet-B0 (frame-level)": "EffNet-B0",
         "R(2+1)D-18": "R(2+1)D-18", "VideoMAE-B": "VideoMAE-B", "DINOv2 linear probe": "DINOv2\nprobe", "CLIP linear probe (UnivFD)": "CLIP probe\n(UnivFD)", "Proposed": "Proposed"}
cols = [("clean", "Clean", BLUE), ("jpeg70", "JPEG q=70", ORANGE), ("down0.25", "Down-sample ×0.25", AQUA), ("blur1.0", "Gaussian blur σ=1", YELLOW)]
fig, ax = plt.subplots(figsize=(10, 3.6))
x = np.arange(len(T2)); w = 0.2
for i, (c, lab, col) in enumerate(cols):
    ax.bar(x + (i - 1.5) * w, T2[c], w * 0.9, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels([names[m] for m in T2.model], fontsize=8.5)
ax.set_ylim(0.6, 1.0); ax.set_ylabel("Accuracy"); ax.legend(ncol=4, frameon=False, loc="upper left", fontsize=8.5)
ax.grid(axis="x", visible=False)
save(fig, "fig5_robustness.png")

# ------------------------------------------------------------- Fig. 6: LODO
T4 = pd.read_csv(os.path.join(P, "t_lodo.csv"))
order = [d for d in T4.disaster.unique() if d != "Pooled"] + ["Pooled"]
fig, ax = plt.subplots(figsize=(9.5, 3.4))
LM = [("ResNet50-SE-BiLSTM-MHA", GREY), ("DINOv2 linear probe", YELLOW), ("CLIP linear probe (UnivFD)", AQUA), ("Proposed", BLUE)]
x = np.arange(len(order)); w = 0.2
for i, (m, col) in enumerate(LM):
    v = T4[T4.model == m].set_index("disaster").loc[order]
    off = (i - 1.5) * w
    ax.bar(x + off, v.acc, w * 0.9, color=col, label=m)
    if m == "Proposed":
        for xi, a in zip(x, v.acc):
            ax.text(xi + off, a + 0.004, f"{a:.2f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([{"Water": "Flood", "Volcanic": "Volcanic"}.get(d, d) for d in order])
ax.set_ylim(0.70, 1.08); ax.set_ylabel("Accuracy on held-out disaster type"); ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=8)
ax.grid(axis="x", visible=False)
save(fig, "fig6_lodo.png")

# ------------------------------------------------------------- Fig. 7: kapı + kol kapatma
R = pd.read_csv(os.path.join(EX, "explain.csv"))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
K = [("none", "Clean"), ("jpeg70", "JPEG 70"), ("down0.25", "Down ×0.25"), ("blur1.0", "Blur σ=1")]
x = np.arange(len(K)); w = 0.36
for i, (yv, lab, col) in enumerate([(0, "Fake", ORANGE), (1, "Real", BLUE)]):
    g = R[R.y == yv]
    m = [g[f"gate_tmp_{k}"].mean() for k, _ in K]; se = [g[f"gate_tmp_{k}"].std() / np.sqrt(len(g)) * 1.96 for k, _ in K]
    a1.bar(x + (i - 0.5) * w, m, w * 0.92, yerr=se, color=col, label=lab, capsize=2, error_kw=dict(lw=0.7))
a1.set_xticks(x); a1.set_xticklabels([l for _, l in K]); a1.set_ylim(0.3, 0.47); a1.set_ylabel("Mean gate weight of temporal stream")
a1.legend(frameon=False, ncol=2, loc="upper left"); a1.grid(axis="x", visible=False); a1.set_title("(a) Gate re-weighting under distortion", fontsize=9)
from sklearn.metrics import accuracy_score
acc = [accuracy_score(R.y, R[c] >= R.thr) for c in ["prob_none", "prob_drop_tmp", "prob_drop_sem"]]
a2.bar([0, 1, 2], acc, 0.6, color=[BLUE, ORANGE, GREY])
for i, a in enumerate(acc):
    a2.text(i, a + 0.003, f"{a:.3f}", ha="center", fontsize=8)
a2.set_xticks([0, 1, 2]); a2.set_xticklabels(["Full model", "Temporal\nremoved", "Semantic\nremoved"]); a2.set_ylim(0.85, 1.0)
a2.set_ylabel("Accuracy"); a2.grid(axis="x", visible=False); a2.set_title("(b) Stream knock-out (clean input)", fontsize=9)
fig.tight_layout(); save(fig, "fig7_gates.png")

# ------------------------------------------------------------- Fig. 8: Grad-CAM
C = np.load(os.path.join(EX, "cams.npz"))
R["correct"] = (R.prob_none >= R.thr) == (R.y == 1); R["conf"] = (R.prob_none - 0.5).abs()


def ov(img, cam):
    cam = cv2.resize(cam.astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    hm = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (0.55 * img + 0.45 * hm).astype(np.uint8)


def cam_fig(rows, fname, t=7, labels=None):
    n = len(rows)
    fig, ax = plt.subplots(n, 3, figsize=(5.2, 1.75 * n))
    for r, (_, row) in enumerate(rows.iterrows()):
        f = frame(row.id, t=t); cs, ct = C[f"{row.id}_sem"][t], C[f"{row.id}_tmp"][t]
        for j, im in enumerate([f, ov(f, cs), ov(f, ct)]):
            ax[r, j].imshow(im); ax[r, j].set_xticks([]); ax[r, j].set_yticks([]); ax[r, j].grid(False)
        lab = labels[r] if labels else ""
        ax[r, 0].set_ylabel(f"{lab}{ {'Water': 'Flood'}.get(row.disaster, row.disaster)}\n{'real' if row.y == 1 else 'fake'}, p(real)={row.prob_none:.2f}", fontsize=7.5)
    for j, tt in enumerate(["Frame", "Semantic (CLIP) Grad-CAM", "Temporal Grad-CAM"]):
        ax[0, j].set_title(tt, fontsize=8)
    fig.tight_layout(pad=0.25); save(fig, fname)


ok = R[R.correct & (R.prob_none.between(0.0, 1.0))]
dark = np.array([np.load(os.path.join(CACHE, "clean_pairs", v + ".npy"), mmap_mode="r")[14].mean() < 25 for v in ok.id])
ok = ok[~dark]
pick = []
for yv, ds in [(0, ["Earthquake", "Water", "Wildfire"]), (1, ["Volcanic", "Earthquake", "Wildfire"])]:
    for d in ds:
        pick.append(ok[(ok.y == yv) & (ok.disaster == d)].sort_values("conf", ascending=False).iloc[0])
cam_fig(pd.DataFrame(pick), "fig8_gradcam.png")
err = R[~R.correct].sort_values("conf", ascending=False).iloc[[0, 1, 5]]
cam_fig(err, "fig9_errors.png")

# ------------------------------------------------------------- Fig. 10: doğruluk-maliyet
B = pd.read_csv(os.path.join(RESULTS, "bench.csv"))
T1 = pd.read_csv(os.path.join(P, "t_main.csv")).set_index("model")
bmap = {"Orig (ResNet50+SE+BiLSTM+MHA)": "ResNet50-SE-BiLSTM-MHA", "Xception (frame)": "Xception (frame-level)",
        "EfficientNet-B0 (frame)": "EfficientNet-B0 (frame-level)", "R(2+1)D-18": "R(2+1)D-18", "VideoMAE-B": "VideoMAE-B",
        "CLIP probe (UnivFD)": "CLIP linear probe (UnivFD)", "Proposed: CLIP sem + temporal": "Proposed"}
fig, ax = plt.subplots(figsize=(6, 3.4))
for bn, tn in bmap.items():
    b = B[B.model == bn].iloc[0]; rr = T2.set_index("model").loc[tn]
    col = BLUE if tn == "Proposed" else DGREY
    ax.scatter(b.latency_ms, rr.mean_pert, s=40 if tn == "Proposed" else 26, color=col, zorder=3, edgecolor="white", lw=1.5)
    off = {"R(2+1)D-18": (-20, 7), "VideoMAE-B": (6, 1), "ResNet50-SE-BiLSTM-MHA": (6, -9), "Proposed": (-14, 7),
           "CLIP linear probe (UnivFD)": (-45, -13)}.get(tn, (6, -3))
    ax.annotate(names[tn].replace("SE-\n", "SE-").replace("\n", " "), (b.latency_ms, rr.mean_pert), xytext=off,
                textcoords="offset points", fontsize=7.5, color="#0b0b0b")
ax.set_xscale("log"); ax.set_xlabel("Inference latency per video (ms, RTX 4090, FP16, log scale)")
ax.set_ylabel("Mean accuracy under distortion"); ax.set_xlim(3, 60)
from matplotlib.ticker import FixedLocator, NullFormatter, FixedFormatter
ax.xaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 40])); ax.xaxis.set_major_formatter(FixedFormatter(["3", "5", "10", "20", "40"]))
ax.xaxis.set_minor_formatter(NullFormatter())
save(fig, "fig10_cost.png")
print("tamam", FIG)

# ------------------------------------------------------------- Fig. R: çözünürlük kontrolü (hakem M1)
R = pd.read_csv(os.path.join(P, "eff_resolution.csv"))
RS = json.load(open(os.path.join(P, "review_stats.json")))["harmonized"]
MT = json.load(open(os.path.join(P, "res_matched.json")))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.3), gridspec_kw=dict(width_ratios=[1, 1.15]))
bins = np.geomspace(140, 1800, 28)
a1.hist(R[R.y == 1].eff_short, bins, color=ORANGE, alpha=0.75, label="Real")
a1.hist(R[R.y == 0].eff_short, bins, color=BLUE, alpha=0.75, label="Generated")
a1.axvline(288, color=DGREY, ls="--", lw=1); a1.text(295, a1.get_ylim()[1] * 0.92, "harmonization\ntarget 288 px", fontsize=7, color=DGREY)
a1.set_xscale("log"); a1.set_xticks([150, 300, 600, 1200]); a1.set_xticklabels(["150", "300", "600", "1200"])
a1.set_xlabel("Effective content short side before resizing (px)"); a1.set_ylabel("Videos"); a1.legend(frameon=False)
a1.set_title("(a) Source resolution by class", fontsize=10, loc="left"); a1.grid(axis="x", visible=False)
cats = ["All videos,\nnative", "Harmonized,\ntest only", "Harmonized,\nretrained", "Resolution-\nmatched subset"]
vals = {"Resolution-only RF": [MT["res_rf_acc_all"], np.nan, np.nan, MT["res_rf_acc"]],
        "CLIP linear probe": [RS["clip_probe"]["clean_acc"], RS["clip_probe"]["harm_acc"], RS["retrained"]["probe_rt_acc"], MT["CLIP linear probe (UnivFD)"]],
        "Proposed": [RS["proposed"]["clean_acc"], RS["proposed"]["harm_acc"], RS["retrained"]["acc"], MT["Proposed"]]}
x = np.arange(4); w = 0.26
for i, ((m, v), col) in enumerate(zip(vals.items(), [GREY, AQUA, BLUE])):
    a2.bar(x + (i - 1) * w, v, w * 0.9, color=col, label=m)
    for xi, a in zip(x, v):
        if not np.isnan(a):
            a2.text(xi + (i - 1) * w, a + 0.008, f"{a:.3f}", ha="center", fontsize=6)
a2.set_xticks(x); a2.set_xticklabels(cats, fontsize=8); a2.set_ylim(0.45, 1.08); a2.set_ylabel("Accuracy")
a2.legend(frameon=False, fontsize=7, ncol=3, loc="upper center"); a2.grid(axis="x", visible=False)
a2.set_title("(b) Accuracy under resolution control", fontsize=10, loc="left")
fig.tight_layout(); save(fig, "fig_resolution.png")
