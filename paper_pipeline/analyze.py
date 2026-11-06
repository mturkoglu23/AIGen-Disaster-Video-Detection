"""
Tüm koşuların tahminlerinden makale tabloları:
  T_main        : fold ortalaması ± std + havuzlanmış %95 bootstrap GA
  T_pairwise    : önerilen modele karşı DeLong (AUC) ve McNemar (doğruluk) testleri
  T_disaster    : afet türü bazında doğruluk / AUC
  T_controls    : kısayol kontrolleri — çözünürlük-eşlenmiş alt küme ve meta veri ile çelişen alt küme
  T_robust      : bozulmalara dayanıklılık
  T_lodo        : afet türleri arası genelleme
"""
import os, sys, glob, json, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
                             balanced_accuracy_score, matthews_corrcoef, average_precision_score)
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, ROOT

REF = sys.argv[1] if len(sys.argv) > 1 else "P_tristream"
OUT = os.path.join(RESULTS, "tables")
os.makedirs(OUT, exist_ok=True)
split = pd.read_csv(os.path.join(RESULTS, "splits.csv"))
meta = pd.read_csv(os.path.join(ROOT, "analysis", "video_metadata.csv"))
split["file"] = split.rel.str.split("/").str[-1]
split = split.merge(meta[["label", "disaster", "file", "width", "height"]], on=["label", "disaster", "file"], how="left")
split["res"] = split.width.astype(int).astype(str) + "x" + split.height.astype(int).astype(str)

runs = {}
for p in sorted(glob.glob(os.path.join(RESULTS, "runs", "*", "preds.csv"))):
    name = os.path.basename(os.path.dirname(p))
    d = pd.read_csv(p)
    if name.startswith("A_") or name.startswith("B_") or name.startswith("L_") or len(d) == len(split):
        runs[name] = d.merge(split[["id", "disaster", "res", "group"]], on="id", how="left")


def m_all(y, p, thr=0.5):
    q = (p >= thr).astype(int)
    return dict(Acc=accuracy_score(y, q), Prec=precision_score(y, q, zero_division=0), Rec=recall_score(y, q),
                F1=f1_score(y, q), AUC=roc_auc_score(y, p), AP=average_precision_score(y, p),
                BalAcc=balanced_accuracy_score(y, q), MCC=matthews_corrcoef(y, q))


def ece(y, p, bins=10):
    e = 0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        s = (p >= lo) & (p < lo + 1 / bins)
        if s.any():
            e += s.mean() * abs(p[s].mean() - y[s].mean())
    return e


def boot_ci(y, p, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = []
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n):
        s = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        v.append(fn(y[s], p[s]))
    return np.percentile(v, [2.5, 97.5])


# ---- DeLong (Sun & Xu 2014 hızlı uygulama)
def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1; i = k
    out = np.empty(n); out[j] = t
    return out


def delong(y, p1, p2):
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v01, v10 = [], [], []
    for p in (p1, p2):
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(np.concatenate([p[pos], p[neg]]))
        auc = tz[:m].sum() / m / n - (m + 1) / 2 / n
        aucs.append(auc)
        v01.append((tz[:m] - tx) / n)
        v10.append(1 - (tz[m:] - ty) / m)
    s01, s10 = np.cov(np.array(v01)), np.cov(np.array(v10))
    s = s01 / m + s10 / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0
    return aucs[0] - aucs[1], 2 * stats.norm.sf(abs(z))


def mcnemar(y, p1, p2):
    c1, c2 = (p1 >= .5) == y, (p2 >= .5) == y
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    return b, c, stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0


# ---------------------------------------------------------------- T_main
rows = []
for name, d in runs.items():
    if name.startswith("L_"):
        continue
    y, p = d.y.values, d.prob.values
    fm = d.groupby("fold").apply(lambda g: pd.Series(m_all(g.y.values, g.prob.values)), include_groups=False)
    r = dict(run=name, n=len(d))
    for k in ["Acc", "Prec", "Rec", "F1", "AUC", "AP", "BalAcc", "MCC"]:
        r[k] = f"{fm[k].mean():.4f} ± {fm[k].std():.4f}"
    lo, hi = boot_ci(y, p, lambda a, b: accuracy_score(a, b >= .5))
    r["Acc_95CI(pooled)"] = f"[{lo:.4f}, {hi:.4f}]"
    lo, hi = boot_ci(y, p, roc_auc_score)
    r["AUC_95CI(pooled)"] = f"[{lo:.4f}, {hi:.4f}]"
    r["ECE"] = round(ece(y, p), 4)
    r["Acc_valthr"] = round(accuracy_score(y, p >= d.thr.values), 4)
    rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "T_main.csv"), index=False)
print("\n=== T_main"); print(T.to_string(index=False))

# ---------------------------------------------------------------- T_pairwise
if REF in runs:
    ref = runs[REF].set_index("id")
    rows = []
    for name, d in runs.items():
        if name == REF or name.startswith(("L_", "A_", "B_")):
            continue
        d = d.set_index("id").loc[ref.index]
        y = ref.y.values
        dauc, pd_ = delong(y, ref.prob.values, d.prob.values)
        b, c, pm = mcnemar(y, ref.prob.values, d.prob.values)
        rows.append(dict(vs=name, dAUC=round(dauc, 4), DeLong_p=f"{pd_:.2e}", ref_only_correct=b, other_only_correct=c, McNemar_p=f"{pm:.2e}"))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "T_pairwise.csv"), index=False)
    print(f"\n=== T_pairwise (ref={REF})"); print(T.to_string(index=False))

# ---------------------------------------------------------------- T_disaster
rows = []
for name, d in runs.items():
    if name.startswith("L_"):
        continue
    r = dict(run=name)
    for dis, g in d.groupby("disaster"):
        r[dis] = f"{accuracy_score(g.y, g.prob >= .5):.3f}/{roc_auc_score(g.y, g.prob):.3f}"
    rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "T_disaster_acc_auc.csv"), index=False)
print("\n=== T_disaster (Acc/AUC)"); print(T.to_string(index=False))

# ---------------------------------------------------------------- T_controls
md = runs.get("Bl_metadata_rf")
if md is not None:
    md = md.set_index("id")
    res_counts = split.groupby(["res", "label"]).size().unstack(fill_value=0)
    matched_res = res_counts[(res_counts.fake >= 5) & (res_counts.real >= 5)].index.tolist()
    print("\nçözünürlük-eşlenmiş çözünürlükler:", res_counts.loc[matched_res].to_dict("index"))
    conflict_ids = md.index[((md.prob >= .5).astype(int) != md.y)]
    unsure_ids = md.index[(md.prob - .5).abs() < .3]
    rows = []
    for name, d in runs.items():
        if name.startswith(("L_", "A_", "B_")):
            continue
        d = d.set_index("id")
        r = dict(run=name)
        for sub, ids in [("res_matched", d.index[d.res.isin(matched_res)]),
                         ("meta_wrong", conflict_ids), ("meta_uncertain", unsure_ids)]:
            g = d.loc[d.index.intersection(ids)]
            r[f"{sub}_n"] = len(g)
            r[f"{sub}_acc"] = round(accuracy_score(g.y, g.prob >= .5), 4)
            r[f"{sub}_auc"] = round(roc_auc_score(g.y, g.prob), 4) if g.y.nunique() > 1 else np.nan
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "T_controls.csv"), index=False)
    print("\n=== T_controls"); print(T.to_string(index=False))

# ---------------------------------------------------------------- T_robust
rows = []
for name, d in runs.items():
    cols = [c for c in d.columns if c.startswith("prob_")]
    if not cols:
        continue
    r = dict(run=name, clean=f"{accuracy_score(d.y, d.prob >= .5):.4f}/{roc_auc_score(d.y, d.prob):.4f}")
    for c in cols:
        r[c[5:]] = f"{accuracy_score(d.y, d[c] >= .5):.4f}/{roc_auc_score(d.y, d[c]):.4f}"
    rows.append(r)
if rows:
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "T_robust_acc_auc.csv"), index=False)
    print("\n=== T_robust (Acc/AUC)"); print(T.to_string(index=False))

# ---------------------------------------------------------------- T_lodo
rows = []
for name, d in runs.items():
    if not name.startswith("L_"):
        continue
    for dis, g in d.groupby("disaster"):
        rows.append(dict(run=name, held_out=dis, n=len(g), Acc=accuracy_score(g.y, g.prob >= .5),
                         AUC=roc_auc_score(g.y, g.prob), F1=f1_score(g.y, g.prob >= .5)))
if rows:
    T = pd.DataFrame(rows).pivot(index="held_out", columns="run", values=["Acc", "AUC"]).round(4)
    T.to_csv(os.path.join(OUT, "T_lodo.csv"))
    print("\n=== T_lodo"); print(T.to_string())

# ---------------------------------------------------------------- akış kapı ağırlıkları
d = runs.get(REF)
if d is not None and any(c.startswith("gate_") for c in d.columns):
    g = d.groupby(["disaster", "y"])[[c for c in d.columns if c.startswith("gate_")]].mean().round(3)
    g.to_csv(os.path.join(OUT, "T_gates.csv"))
    print("\n=== gate weights"); print(g.to_string())
