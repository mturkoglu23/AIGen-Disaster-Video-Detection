"""explain.py çıktılarından özet tablolar ve Grad-CAM şekilleri.  python explain_fig.py --tag P_clip_semtmp_s1"""
import os, sys, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, RESULTS

p = argparse.ArgumentParser(); p.add_argument("--tag", default="P_clip_semtmp_s1"); args = p.parse_args()
OUT = os.path.join(RESULTS, "explain", args.tag)
R = pd.read_csv(os.path.join(OUT, "explain.csv"))
C = np.load(os.path.join(OUT, "cams.npz"))
R["cls"] = np.where(R.y == 1, "real", "fake")
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

# ---------------- 1) kapı ağırlıkları
KINDS = ["none", "jpeg70", "down0.25", "blur1.0"]
P("## Kapı ağırlığı (gate_tmp; gate_sem = 1 - gate_tmp), ortalama ± std")
g = pd.DataFrame({k: R[f"gate_tmp_{k}"] for k in KINDS}); g["cls"] = R.cls
P(g.groupby("cls")[KINDS].agg(lambda s: f"{s.mean():.3f}±{s.std():.3f}").to_string())
P("tümü:", " ".join(f"{k}={R[f'gate_tmp_{k}'].mean():.3f}" for k in KINDS))
P("\n## Afet türüne göre gate_tmp (temiz)")
P(R.groupby(["disaster", "cls"]).gate_tmp_none.mean().unstack().round(3).to_string())
R["correct"] = (R.prob_none >= R.thr) == (R.y == 1)
P("\n## Doğru/yanlış tahminlerde gate_tmp:", R.groupby("correct").gate_tmp_none.mean().round(3).to_dict())

# ---------------- 2) kol kapatma
P("\n## Kol kapatma (temiz girdi, fold eşiği)")
for c, name in [("prob_none", "tam model"), ("prob_drop_tmp", "zamansal kapalı"), ("prob_drop_sem", "anlamsal kapalı")]:
    pr = (R[c] >= R.thr).astype(int)
    P(f"{name:16s} acc={accuracy_score(R.y, pr):.4f} f1={f1_score(R.y, pr):.4f}  "
      f"|Δp| ort={np.abs(R[c] - R.prob_none).mean():.3f}")

# ---------------- 3) kenar bandı payı
P("\n## Grad-CAM enerjisinin dış %15 kenar bandındaki payı (düzgün dağılım = 0.510)")
P(R.groupby("cls")[["border_sem", "border_tmp"]].mean().round(3).to_string())
open(os.path.join(OUT, "summary.txt"), "w", encoding="utf8").write("\n".join(lines))

# ---------------- 4) şekiller
def frames(vid):
    x = np.load(os.path.join(CACHE, "clean_pairs", vid + ".npy"))
    return x[0::2]   # 16 konum (t)

def overlay(img, cam):
    cam = cv2.resize(cam.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    hm = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (0.55 * img + 0.45 * hm).astype(np.uint8)

POS = [0, 7, 15]
def figure(sel, fname, title):
    n = len(sel)
    fig, ax = plt.subplots(n, 1 + 2 * len(POS), figsize=(2 * (1 + 2 * len(POS)), 2.05 * n))
    ax = np.atleast_2d(ax)
    for r, (_, row) in enumerate(sel.iterrows()):
        fr = frames(row.id); cs = C[f"{row.id}_sem"]; ct = C[f"{row.id}_tmp"]
        ax[r, 0].imshow(fr[POS[0]])
        ax[r, 0].set_ylabel(f"{row.disaster}\n{row.cls}  p(real)={row.prob_none:.2f}", fontsize=7)
        for j, t in enumerate(POS):
            ax[r, 1 + 2 * j].imshow(overlay(fr[t], cs[t])); ax[r, 2 + 2 * j].imshow(overlay(fr[t], ct[t]))
            if r == 0:
                ax[r, 1 + 2 * j].set_title(f"CLIP t={t}", fontsize=8); ax[r, 2 + 2 * j].set_title(f"Temporal t={t}", fontsize=8)
        if r == 0:
            ax[r, 0].set_title("frame", fontsize=8)
        for a in ax[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130); plt.close(fig)

R["conf"] = np.abs(R.prob_none - 0.5)
for cls in ["fake", "real"]:
    s = R[R.correct & (R.cls == cls)].sort_values("conf", ascending=False).groupby("disaster").head(1).sort_values("disaster")
    figure(s, f"gradcam_{cls}.jpg", f"Grad-CAM, correctly classified {cls} videos (one per disaster type)")
err = R[~R.correct].sort_values("conf", ascending=False).head(6)
if len(err):
    figure(err, "gradcam_errors.jpg", "Grad-CAM, most confident errors")
print("şekiller:", OUT)
