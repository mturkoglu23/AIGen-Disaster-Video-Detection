"""Kopya / aynı-kaynak gruplarını oluşturur (bağlı bileşenler) ve grup bazlı 5-fold bölmeyi kaydeder."""
import os, sys, json, numpy as np, pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, RESULTS
df = pd.read_csv(os.path.join(CACHE, "videos.csv"))
F = np.load(os.path.join(RESULTS, "dino_frame_match.npy")); S = np.load(os.path.join(RESULTS, "dino_video_sim.npy"))
N = len(df); y = df.y.values
md5 = pd.read_csv(os.path.join(RESULTS, "md5.csv")) if os.path.exists(os.path.join(RESULTS, "md5.csv")) else None
for fm, vs in [(0.25, 0.90), (0.25, 0.88), (0.5, 0.93)]:
    A = ((F >= fm) | (S >= vs)) & (y[:, None] == y[None, :])
    k, lab = connected_components(csr_matrix(A), directed=False)
    sz = np.bincount(lab)
    print(f"fm>={fm} | vsim>={vs}: {k} grup, tekil olmayan grup={np.sum(sz>1)}, gruplardaki video={sz[sz>1].sum()}, en büyük={sorted(sz)[-5:]}")
# seçilen eşik
A = ((F >= 0.25) | (S >= 0.90)) & (y[:, None] == y[None, :])
k, lab = connected_components(csr_matrix(A), directed=False)
df["group"] = lab
sz = df.group.map(df.group.value_counts())
print(pd.crosstab(df.label, sz > 1, margins=True))
# exact duplicates (aynı md5) — ikinci kopyayı çıkar
import hashlib
def md5f(p):
    h = hashlib.md5(open(p, "rb").read()).hexdigest(); return h
df["md5"] = [md5f(p) for p in df.path]
df["drop_exact_dup"] = df.duplicated("md5", keep="first")
print("çıkarılan birebir kopya:", df.drop_exact_dup.sum(), df[df.drop_exact_dup].rel.tolist())
d = df[~df.drop_exact_dup].reset_index(drop=True)
sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=42)
d["fold"] = -1
for f, (_, te) in enumerate(sgkf.split(d, d.y, d.group)):
    d.loc[te, "fold"] = f
print(pd.crosstab(d.fold, d.label))
# rastgele (gruplamasız) split'te kaç test videosunun eğitimde aynı-kaynak eşi var?
from sklearn.model_selection import StratifiedKFold
leak = 0
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(d, d.y):
    g_tr = set(d.group.values[tr]); leak += sum(d.group.values[i] in g_tr for i in te)
print(f"Rastgele 5-fold'da eğitimde aynı-kaynak eşi olan test videosu: {leak} / {len(d)} ({leak/len(d):.1%})")
d.drop(columns=["path"]).to_csv(os.path.join(RESULTS, "splits.csv"), index=False)
