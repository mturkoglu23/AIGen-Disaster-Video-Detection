"""Hakem M1: çözünürlük kısayolu kontrolü.
Her video için etkin içerik kısa kenarı (bant + %10 kenar kırpımı sonrası, 224'e ölçeklemeden önce) hesaplanır;
doğruluk sınıf x çözünürlük kutusuna göre ve çözünürlüğü eşleştirilmiş alt kümede raporlanır.
python res_analysis.py  ->  results/paper/t_resolution.csv, results/paper/res_matched.json
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, RESULTS

sp = pd.read_csv(os.path.join(RESULTS, "splits.csv")).set_index("id")
rows = []
for vid in sp.index:
    m = json.load(open(os.path.join(CACHE, "meta", vid + ".json")))
    t, b, l, r = m["rect"]
    h, w = m["height"] * (1 - t - b), m["width"] * (1 - l - r)
    rows.append(dict(id=vid, eff_short=0.8 * min(h, w), native_short=min(m["height"], m["width"])))
R = pd.DataFrame(rows).set_index("id").join(sp[["y", "disaster"]])
R.to_csv(os.path.join(RESULTS, "paper", "eff_resolution.csv"))
print(R.groupby("y").eff_short.describe().round(0))

MODELS = [("Proposed", "Ab_clip_sem_tmp_R"), ("CLIP linear probe (UnivFD)", "Bl_clip_univfd_probe"),
          ("DINOv2 linear probe", "Bl_dinov2_probe"), ("VideoMAE-B", "Bl_videomae"), ("Xception (frame-level)", "Bl_xception"),
          ("ResNet50-SE-BiLSTM-MHA", "D_orig_clean_group_fixed"), ("Metadata RF (8 features)", "Bl_metadata_rf"),
          ("Metadata RF (aspect ratio only)", "Bl_metadata_rf_aspect")]
BINS = [0, 300, 450, 1e9]
LAB = ["<300 px", "300-450 px", ">=450 px"]
R["bin"] = pd.cut(R.eff_short, BINS, labels=LAB, right=False)

# Çözünürlük-eşleştirilmiş alt küme: her iki sınıfın yeterince bulunduğu kutu(lar)
# 1) etkin kısa kenar ikili sınıflandırıcısı (tek öznitelikle kısayol gücü)
from sklearn.ensemble import RandomForestClassifier
res_rf = np.zeros(len(R))
for k in range(5):
    te = sp.loc[R.index, "fold"].values == k
    clf = RandomForestClassifier(500, random_state=42, n_jobs=-1).fit(R[["eff_short"]].values[~te], R.y.values[~te])
    res_rf[te] = clf.predict(R[["eff_short"]].values[te])
R["res_rf"] = res_rf
print("resolution-only RF acc:", round(accuracy_score(R.y, res_rf), 4))

# 2) eşleştirilmiş alt küme: her sınıftan, diğer sınıfın etkin kısa kenar aralığı (5.-95. yüzdelik) içinde kalanlar
lo = max(R[R.y == 0].eff_short.quantile(.05), R[R.y == 1].eff_short.quantile(.05))
hi = min(R[R.y == 0].eff_short.quantile(.95), R[R.y == 1].eff_short.quantile(.95))
# daha sıkı: bire-bir en yakın komşu eşleştirmesi (yerine koymadan) sahte <-> gerçek
fk = R[R.y == 0].sort_values("eff_short"); rl = R[R.y == 1].copy()
pairs = []
used = set()
for vid, row in fk.iterrows():
    cand = rl[~rl.index.isin(used)]
    j = (cand.eff_short - row.eff_short).abs().idxmin()
    if abs(cand.loc[j, "eff_short"] - row.eff_short) <= 0.05 * row.eff_short:
        pairs += [vid, j]; used.add(j)
M = R.loc[pairs]
print(f"matched subset: {len(M)} videos ({(M.y==0).sum()} fake / {(M.y==1).sum()} real), "
      f"eff_short fake median {M[M.y==0].eff_short.median():.0f} real {M[M.y==1].eff_short.median():.0f}")
print("resolution-only RF on matched:", round(accuracy_score(M.y, M.res_rf), 4))

out, matched = [], {"n": len(M), "n_fake": int((M.y == 0).sum()), "n_real": int((M.y == 1).sum()),
                    "res_rf_acc": accuracy_score(M.y, M.res_rf), "res_rf_acc_all": accuracy_score(R.y, R.res_rf)}
for name, tag in MODELS:
    s = pd.read_csv(os.path.join(RESULTS, "runs", tag, "preds.csv")).set_index("id").loc[R.index]
    c = ((s.prob >= s.thr).astype(int) == s.y)
    d = dict(model=name, all=c.mean())
    for y in (0, 1):
        for bl in LAB:
            sel = (R.y == y) & (R.bin == bl)
            d[f"{'fake' if y == 0 else 'real'} {bl}"] = c[sel].mean() if sel.sum() else np.nan
    d["matched"] = c.loc[M.index].mean()
    matched[name] = d["matched"]
    out.append(d)
T = pd.DataFrame(out)
cnt = {f"{'fake' if y == 0 else 'real'} {bl}": int(((R.y == y) & (R.bin == bl)).sum()) for y in (0, 1) for bl in LAB}
T = pd.concat([pd.DataFrame([dict(model="n", all=len(R), matched=len(M), **cnt)]), T])
T.to_csv(os.path.join(RESULTS, "paper", "t_resolution.csv"), index=False)
json.dump(matched, open(os.path.join(RESULTS, "paper", "res_matched.json"), "w"), indent=1)
print(T.round(3).to_string(index=False))
