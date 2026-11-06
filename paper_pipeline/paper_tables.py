"""Makale tabloları (birleşik/pooled metrikler, McNemar + eşli bootstrap F1).  python paper_tables.py"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import *
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
src = open(os.path.join(os.path.dirname(__file__), "analyze.py"), encoding="utf8").read()
exec(src[src.index("def _midrank"):src.index("# ----------------------------------------------------------------")])
OUT = os.path.join(RESULTS, "paper"); os.makedirs(OUT, exist_ok=True)
REF = "Ab_clip_sem_tmp_R"
PERT = ["jpeg70", "down0.25", "blur1.0"]


def load(t):
    return pd.read_csv(os.path.join(RESULTS, "runs", t, "preds.csv")).set_index("id")


ref = load(REF)
ids = ref.index
rng = np.random.default_rng(0)
BOOT = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]


def pred(s, col="prob"):
    return (s[col] >= s.thr).astype(int).values


def cmp(s, col):
    y = ref.y.values; p1 = pred(ref, col); p2 = pred(s, col)
    _, _, pa = mcnemar(y, p1, p2)
    d = np.array([f1_score(y[i], p1[i]) - f1_score(y[i], p2[i]) for i in BOOT])
    pf = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return pa, pf


def fold_std(s, col="prob"):
    return np.std([accuracy_score(g.y, g[col] >= g.thr) for _, g in s.groupby("fold")], ddof=1)


# ---------------------------------------------------------------- Tablo: ana karşılaştırma
MAIN = [("Metadata RF (8 container features)", "Bl_metadata_rf"),
        ("ResNet50-SE-BiLSTM-MHA", "D_orig_clean_group_fixed"),
        ("Xception (frame-level)", "Bl_xception"),
        ("EfficientNet-B0 (frame-level)", "Bl_effb0"),
        ("R(2+1)D-18", "Bl_r2plus1d"),
        ("VideoMAE-B", "Bl_videomae"),
        ("DINOv2 linear probe", "Bl_dinov2_probe"),
        ("CLIP linear probe (UnivFD)", "Bl_clip_univfd_probe"),
        ("Proposed", REF)]
rows = []
for name, t in MAIN:
    s = load(t).loc[ids]; y = s.y.values; p = pred(s)
    r = dict(model=name, acc=accuracy_score(y, p), acc_sd=fold_std(s), prec=precision_score(y, p), rec=recall_score(y, p),
             f1=f1_score(y, p), mcc=matthews_corrcoef(y, p), bal=balanced_accuracy_score(y, p), auc=roc_auc_score(y, s.prob))
    if t != REF:
        r["p_acc"], r["p_f1"] = cmp(s, "prob")
    rows.append(r)
T1 = pd.DataFrame(rows); T1.to_csv(os.path.join(OUT, "t_main.csv"), index=False)
print(T1.round(4).to_string(index=False))

# ---------------------------------------------------------------- Tablo: sağlamlık
rows = []
for name, t in MAIN[1:]:
    s = load(t).loc[ids]; y = s.y.values
    r = dict(model=name, clean=accuracy_score(y, pred(s)))
    for k in PERT:
        r[k] = accuracy_score(y, pred(s, f"prob_{k}")); r[k + "_f1"] = f1_score(y, pred(s, f"prob_{k}"))
        if t != REF:
            r[k + "_p"] = cmp(s, f"prob_{k}")[0]
    r["mean_pert"] = np.mean([r[k] for k in PERT]); r["drop"] = r["clean"] - r["mean_pert"]
    rows.append(r)
T2 = pd.DataFrame(rows); T2.to_csv(os.path.join(OUT, "t_robust.csv"), index=False)
print(T2.round(4).to_string(index=False))

# ---------------------------------------------------------------- Tablo: ablasyon
ABL = [("Semantic (CLIP)", "Ab_sem_clip"), ("Forensic (SRM)", "Ab_for_R"), ("Temporal", "Ab_tmp_R"),
       ("Semantic + forensic", "Ab_clip_sem_for_R"), ("Forensic + temporal", "Ab_for_tmp_R"),
       ("Semantic + temporal (proposed)", REF), ("Semantic + forensic + temporal", "P_tristream_clip")]
rows = []
for name, t in ABL:
    s = load(t).loc[ids]; y = s.y.values
    r = dict(config=name, clean_acc=accuracy_score(y, pred(s)), clean_f1=f1_score(y, pred(s)))
    if t != REF:
        r["clean_p"] = cmp(s, "prob")[0]
    for k in PERT:
        r[k] = accuracy_score(y, pred(s, f"prob_{k}"))
        if t != REF:
            r[k + "_p"] = cmp(s, f"prob_{k}")[0]
    rows.append(r)
T3 = pd.DataFrame(rows); T3.to_csv(os.path.join(OUT, "t_ablation.csv"), index=False)
print(T3.round(4).to_string(index=False))

# ---------------------------------------------------------------- Tablo: LODO
sp = pd.read_csv(os.path.join(RESULTS, "splits.csv")).set_index("id")
rows = []
for name, t in [("ResNet50-SE-BiLSTM-MHA", "L_orig_fixed"), ("DINOv2 linear probe", "L_dinov2_probe"),
                ("CLIP linear probe (UnivFD)", "L_clip_probe"), ("Proposed", "L_clip_semtmp")]:
    s = load(t).join(sp[["disaster"]])
    for d, g in s.groupby("disaster"):
        p = (g.prob >= g.thr).astype(int)
        rows.append(dict(model=name, disaster=d, n=len(g), n_fake=int((g.y == 0).sum()), acc=accuracy_score(g.y, p), f1=f1_score(g.y, p),
                         auc=roc_auc_score(g.y, g.prob)))
    p = (s.prob >= s.thr).astype(int)
    rows.append(dict(model=name, disaster="Pooled", n=len(s), n_fake=int((s.y == 0).sum()), acc=accuracy_score(s.y, p), f1=f1_score(s.y, p),
                     auc=roc_auc_score(s.y, s.prob)))
T4 = pd.DataFrame(rows); T4.to_csv(os.path.join(OUT, "t_lodo.csv"), index=False)
a, b = load("L_orig_fixed"), load("L_clip_semtmp"); a = a.loc[b.index]
print(T4.round(4).to_string(index=False)); print("LODO McNemar", mcnemar(b.y.values, pred(b), pred(a)))

# ---------------------------------------------------------------- Tablo: protokol / kısayol
rows = []
for name, t in [("Metadata RF, 8 features", "Bl_metadata_rf"), ("Metadata RF, aspect ratio only", "Bl_metadata_rf_aspect"),
                ("ResNet50-SE-BiLSTM-MHA, raw frames, random split, original recipe", "A_orig_raw_random"),
                ("ResNet50-SE-BiLSTM-MHA, raw frames, group split, original recipe", "B_orig_raw_group"),
                ("ResNet50-SE-BiLSTM-MHA, clean frames, group split, original recipe", "C_orig_clean_group"),
                ("ResNet50-SE-BiLSTM-MHA, clean frames, group split, revised recipe", "D_orig_clean_group_fixed")]:
    s = load(t); p = pred(s)
    rows.append(dict(setting=name, acc=accuracy_score(s.y, p), f1=f1_score(s.y, p)))
T5 = pd.DataFrame(rows); T5.to_csv(os.path.join(OUT, "t_shortcut.csv"), index=False)
print(T5.round(4).to_string(index=False))

# ---------------------------------------------------------------- tohumlar
rows = []
for t in [REF, "P_clip_semtmp_s1", "P_clip_semtmp_s2"]:
    s = load(t).loc[ids]; y = s.y.values
    rows.append(dict(run=t, acc=accuracy_score(y, pred(s)), f1=f1_score(y, pred(s)), **{k: accuracy_score(y, pred(s, f"prob_{k}")) for k in PERT}))
T6 = pd.DataFrame(rows); T6.to_csv(os.path.join(OUT, "t_seeds.csv"), index=False)
print(T6.round(4).to_string(index=False)); print(T6.drop(columns="run").agg(["mean", "std"]).round(4).to_string())
