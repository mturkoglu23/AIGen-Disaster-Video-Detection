"""Hakem revizyonu istatistikleri: Holm düzeltmesi, val-eşikli probe'lar, LODO'da CLIP probe, uyumlaştırılmış çözünürlük.
python review_stats.py -> results/paper/review_stats.json (+ ekrana)
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
src = open(os.path.join(os.path.dirname(__file__), "analyze.py"), encoding="utf8").read()
exec(src[src.index("def _midrank"):src.index("# ----------------------------------------------------------------")])
P = os.path.join(RESULTS, "paper")
out = {}


def load(t):
    return pd.read_csv(os.path.join(RESULTS, "runs", t, "preds.csv")).set_index("id")


def pr(s, col="prob"):
    return (s[col] >= s.thr).astype(int).values


def holm(d):
    keys = list(d); p = np.array([d[k] for k in keys], float); o = np.argsort(p); m = len(p)
    q = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); q[i] = run
    return dict(zip(keys, q))


# 1) Holm: ana tablo (7 baseline, doğruluk ve F1) ve sağlamlık (7 baseline x 3 bozulma) aileleri
T1 = pd.read_csv(os.path.join(P, "t_main.csv")).dropna(subset=["p_acc"])
T1 = T1[~T1.model.str.startswith("Metadata")]
out["holm_main_acc"] = holm(dict(zip(T1.model, T1.p_acc)))
out["holm_main_f1"] = holm(dict(zip(T1.model, T1.p_f1)))
T2 = pd.read_csv(os.path.join(P, "t_robust.csv")); T2 = T2[T2.model != "Proposed"]
out["holm_robust"] = holm({f"{m}|{k}": v for m, r in T2.set_index("model").iterrows() for k, v in
                           ((k, r[k + "_p"]) for k in ["jpeg70", "down0.25", "blur1.0"])})
T3 = pd.read_csv(os.path.join(P, "t_ablation.csv")).dropna(subset=["clean_p"])
out["holm_ablation"] = holm({f"{c}|{k}": r[f"{k}_p"] for c, r in T3.set_index("config").iterrows()
                             for k in ["clean", "jpeg70", "down0.25", "blur1.0"]})

# 2) Val-eşikli probe'lar vs önerilen
ref = load("Ab_clip_sem_tmp_R"); ids = ref.index; yy = ref.y.values
for tag in ["Bl_clip_probe_vthr", "Bl_dinov2_probe_vthr"]:
    s = load(tag).loc[ids]; d = {}
    for col in ["prob", "prob_jpeg70", "prob_down0.25", "prob_blur1.0"]:
        d[col] = dict(acc=accuracy_score(yy, pr(s, col)), f1=f1_score(yy, pr(s, col)), p=mcnemar(yy, pr(ref, col), pr(s, col))[2])
    d["auc"] = roc_auc_score(yy, s.prob)
    out[tag] = d

# 3) LODO: önerilen vs CLIP / DINOv2 probe
sp = pd.read_csv(os.path.join(RESULTS, "splits.csv")).set_index("id")
L = load("L_clip_semtmp")
for tag in ["L_clip_probe", "L_dinov2_probe"]:
    s = load(tag).loc[L.index].join(sp[["disaster"]])
    per = {d: dict(acc=accuracy_score(g.y, g.prob >= g.thr), f1=f1_score(g.y, g.prob >= g.thr), auc=roc_auc_score(g.y, g.prob))
           for d, g in s.groupby("disaster")}
    per["Pooled"] = dict(acc=accuracy_score(s.y, pr(s)), f1=f1_score(s.y, pr(s)), auc=roc_auc_score(s.y, s.prob))
    out[tag] = dict(per=per, p_vs_proposed=mcnemar(L.y.values, pr(L), pr(s))[2])

# 4) Uyumlaştırılmış çözünürlük (varsa)
hp = os.path.join(RESULTS, "harm_eval.csv")
if os.path.exists(hp):
    H = pd.read_csv(hp).set_index("id")
    R = pd.read_csv(os.path.join(P, "eff_resolution.csv")).set_index("id")
    d = {}
    for name in [c[5:] for c in H.columns if c.startswith("prob_")]:
        t = H[f"thr_{name}"]
        p_cl = (H[f"clean_{name}"] >= t).astype(int); p_h = (H[f"prob_{name}"] >= t).astype(int)
        hi = R.loc[H.index].eff_short > 288
        f, r = H.y == 0, H.y == 1
        d[name] = dict(clean_acc=accuracy_score(H.y, p_cl), harm_acc=accuracy_score(H.y, p_h), harm_f1=f1_score(H.y, p_h),
                       harm_auc=roc_auc_score(H.y, H[f"prob_{name}"]), fake_clean=p_cl[f].eq(0).mean(), fake_harm=p_h[f].eq(0).mean(),
                       real_clean=p_cl[r].eq(1).mean(), real_harm=p_h[r].eq(1).mean(), n_changed_input=int(hi.sum()),
                       p_clean_vs_harm=mcnemar(H.y.values, p_cl.values, p_h.values)[2])
    d["p_proposed_vs_probe_harm"] = mcnemar(H.y.values, (H.prob_proposed >= H.thr_proposed).astype(int).values,
                                            (H.prob_clip_probe >= H.thr_clip_probe).astype(int).values)[2]
    # yeniden eğitim (uyumlaştırılmış girdiyle sıfırdan eğitilmiş önerilen model)
    hr = os.path.join(RESULTS, "runs", "H_clip_semtmp", "preds.csv")
    if os.path.exists(hr):
        Hm = load("H_clip_semtmp").loc[H.index]; ph = pr(Hm); pa = pr(load("Ab_clip_sem_tmp_R").loc[H.index]); yv = Hm.y.values
        pp = (H.prob_clip_probe_rt >= H.thr_clip_probe_rt).astype(int).values
        d["retrained"] = dict(acc=accuracy_score(yv, ph), f1=f1_score(yv, ph), auc=roc_auc_score(yv, Hm.prob),
                              fake=float((ph[yv == 0] == 0).mean()), real=float((ph[yv == 1] == 1).mean()),
                              p_vs_native=mcnemar(yv, pa, ph)[2], probe_rt_acc=accuracy_score(yv, pp), p_vs_probe_rt=mcnemar(yv, ph, pp)[2],
                              **{k: accuracy_score(yv, pr(Hm, "prob_" + k)) for k in ["jpeg70", "down0.25", "blur1.0"]})
    out["harmonized"] = d

json.dump(out, open(os.path.join(P, "review_stats.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=lambda v: round(float(v), 4)))
