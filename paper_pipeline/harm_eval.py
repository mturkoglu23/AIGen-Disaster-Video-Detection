"""Hakem M1: eğitilmiş modelleri (yeniden eğitmeden) çözünürlük-uyumlaştırılmış girdide test et.
Önerilen model: P_clip_semtmp_s1 kat ağırlıkları + kendi doğrulama eşikleri. CLIP probe: probe_extra.py ile aynı
(val-eşikli) kurulum; test gömmeleri uyumlaştırılmış karelerden.
python harm_eval.py [harm288_pairs] -> results/harm_eval.csv
"""
import os, sys, json
import numpy as np, pandas as pd, torch, timm, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, RESULTS
from models import build_model

DATA = sys.argv[1] if len(sys.argv) > 1 else "harm288_pairs"
dev = "cuda"
df = pd.read_csv(os.path.join(RESULTS, "splits.csv"))
RUN = os.path.join(RESULTS, "runs", "P_clip_semtmp_s1")
A = json.load(open(os.path.join(RUN, "args.json")))
P1 = pd.read_csv(os.path.join(RUN, "preds.csv")).set_index("id")
load = lambda d, ids: torch.from_numpy(np.stack([np.load(os.path.join(CACHE, d, i + ".npy")) for i in ids]))
res = pd.DataFrame(index=df.id); res["y"] = df.y.values; res["fold"] = df.fold.values

# ---- önerilen model
for k in range(5):
    m = build_model("tristream", streams=tuple(A["streams"].split(",")), stream_drop=A["stream_drop"],
                    sem_backbone=A["sem_backbone"], sem_unfreeze_last=0, sem2_backbone=A["sem2_backbone"])
    miss, unexp = m.load_state_dict(torch.load(os.path.join(RUN, f"fold{k}.pt"), map_location="cpu"), strict=False)
    assert not unexp
    m = m.to(dev).eval()
    ids = df.id[df.fold == k].values
    for d, col in [("clean_pairs", "clean_proposed"), (DATA, "prob_proposed")]:
        X, pr = load(d, ids), []
        with torch.no_grad():
            for s in range(0, len(ids), 16):
                with torch.autocast("cuda", dtype=torch.float16):
                    pr.append(torch.sigmoid(m(X[s:s + 16].to(dev))["logit"].float()).cpu())
        res.loc[ids, col] = torch.cat(pr).numpy()
    res.loc[ids, "thr_proposed"] = P1.loc[ids, "thr"].values
    print("fold", k, flush=True)
    del m; torch.cuda.empty_cache()

# ---- CLIP probe (val eşikli)
clip = timm.create_model("vit_large_patch14_clip_224.openai", pretrained=True, num_classes=0).to(dev).eval().half()
cfg = clip.pretrained_cfg
mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1).half(); std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1).half()
E = []
with torch.no_grad():
    for vid in df.id:
        x = torch.from_numpy(np.load(os.path.join(CACHE, DATA, vid + ".npy"))[::2]).to(dev).permute(0, 3, 1, 2).half() / 255
        E.append(F.normalize(clip((x - mean) / std).float(), dim=-1).mean(0).cpu().numpy())
Eh = np.stack(E); np.save(os.path.join(RESULTS, f"clip_emb_{DATA}.npy"), Eh); Ec = np.load(os.path.join(RESULTS, "clip_l14_emb.npy"))
y, grp, GRID = df.y.values, df.group.values, np.arange(0.05, 0.951, 0.01)
for k in range(5):
    te = np.where(df.fold.values == k)[0]; tri = np.where(df.fold.values != k)[0]
    a, b = next(StratifiedGroupKFold(7, shuffle=True, random_state=42 + k).split(tri, y[tri], grp[tri]))
    tr, va = tri[a], tri[b]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)).fit(Ec[tr], y[tr])
    vp = clf.predict_proba(Ec[va])[:, 1]
    thr = float(GRID[np.argmax([f1_score(y[va], vp >= t, zero_division=0) for t in GRID])])
    ids = df.id.values[te]
    res.loc[ids, "clean_clip_probe"] = clf.predict_proba(Ec[te])[:, 1]
    res.loc[ids, "prob_clip_probe"] = clf.predict_proba(Eh[te])[:, 1]
    res.loc[ids, "thr_clip_probe"] = thr
    # aynı kurulumda uyumlaştırılmış gömmelerle eğitilen probe (eğitim ve test aynı alan)
    c2 = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)).fit(Eh[tr], y[tr])
    v2 = c2.predict_proba(Eh[va])[:, 1]
    res.loc[ids, "thr_clip_probe_rt"] = float(GRID[np.argmax([f1_score(y[va], v2 >= t, zero_division=0) for t in GRID])])
    res.loc[ids, "prob_clip_probe_rt"] = c2.predict_proba(Eh[te])[:, 1]
    res.loc[ids, "clean_clip_probe_rt"] = res.loc[ids, "clean_clip_probe"]

res.index.name = "id"
res.to_csv(os.path.join(RESULTS, "harm_eval.csv"))
for n in ["proposed", "clip_probe"]:
    t = res[f"thr_{n}"]
    print(n, "clean", round(accuracy_score(res.y, res[f"clean_{n}"] >= t), 4), "harm", round(accuracy_score(res.y, res[f"prob_{n}"] >= t), 4))
