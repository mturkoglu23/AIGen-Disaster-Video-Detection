"""Hakem M4 + Minor-1: lineer probe'lar için (a) doğrulama kümesinde seçilen eşik (train.py ile aynı
make_val ve F1 ızgarası), (b) leave-one-disaster-out. Probe yalnız eğitim kısmında (val hariç) eğitilir,
tıpkı sinir ağı modelleri gibi.
Çıktılar: runs/Bl_clip_probe_vthr, runs/Bl_dinov2_probe_vthr, runs/L_clip_probe, runs/L_dinov2_probe (preds.csv)
"""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, RESULTS

df = pd.read_csv(os.path.join(RESULTS, "splits.csv"))
KINDS = ["jpeg70", "down0.25", "blur1.0"]
all_ids = pd.read_csv(os.path.join(CACHE, "videos.csv")).id.tolist()
EMB = {}
for key, run in [("clip", "Bl_clip_univfd_probe"), ("dinov2", "Bl_dinov2_probe")]:
    Ep = dict(np.load(os.path.join(RESULTS, f"robust_emb_{run}.npz")))
    EMB[key] = {k: Ep[k] for k in KINDS}
EMB["clip"]["none"] = np.load(os.path.join(RESULTS, "clip_l14_emb.npy"))
EMB["dinov2"]["none"] = np.load(os.path.join(RESULTS, "dino_frame_emb.npy")).astype(np.float32).mean(1)[[all_ids.index(i) for i in df.id]]
y, grp = df.y.values, df.group.values
GRID = np.arange(0.05, 0.951, 0.01)


def make_val(tr_idx, k):
    sg = StratifiedGroupKFold(7, shuffle=True, random_state=42 + k)
    a, b = next(sg.split(tr_idx, y[tr_idx], grp[tr_idx]))
    return tr_idx[a], tr_idx[b]


def run(key, fold_of, tag):
    E = EMB[key]; rows = []
    for k in sorted(set(fold_of)):
        te = np.where(fold_of == k)[0]
        tr, va = make_val(np.where(fold_of != k)[0], k)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)).fit(E["none"][tr], y[tr])
        vp = clf.predict_proba(E["none"][va])[:, 1]
        thr = float(GRID[np.argmax([f1_score(y[va], vp >= t, zero_division=0) for t in GRID])])
        r = dict(id=df.id.values[te], fold=k, y=y[te], prob=clf.predict_proba(E["none"][te])[:, 1], thr=thr)
        for kd in KINDS:
            r[f"prob_{kd}"] = clf.predict_proba(E[kd][te])[:, 1]
        rows.append(pd.DataFrame(r))
    pr = pd.concat(rows)
    out = os.path.join(RESULTS, "runs", tag); os.makedirs(out, exist_ok=True)
    pr.to_csv(os.path.join(out, "preds.csv"), index=False)
    p = (pr.prob >= pr.thr)
    print(f"{tag:22s} acc={accuracy_score(pr.y, p):.4f} f1={f1_score(pr.y, p):.4f} auc={roc_auc_score(pr.y, pr.prob):.4f} "
          + " ".join(f"{kd}={accuracy_score(pr.y, pr[f'prob_{kd}'] >= pr.thr):.4f}" for kd in KINDS))


dis = sorted(df.disaster.unique())
lodo = df.disaster.map({d: i for i, d in enumerate(dis)}).values
for key in ("clip", "dinov2"):
    run(key, df.fold.values, f"Bl_{key}_probe_vthr")
    run(key, lodo, f"L_{key}_probe")
