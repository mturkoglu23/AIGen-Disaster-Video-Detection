"""Kare düzeyinde DINOv2 eşleşmesi ile kopya kalibrasyonu ve görsel kontrol paftası."""
import os, sys, numpy as np, pandas as pd, torch, cv2
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, RESULTS
df = pd.read_csv(os.path.join(CACHE, "videos.csv"))
E = torch.from_numpy(np.load(os.path.join(RESULTS, "dino_frame_emb.npy")).astype(np.float32)).cuda()  # N,16,D
N = len(E); F = torch.zeros(N, N, device="cuda")
for a in range(N):
    s = torch.einsum("fd,ngd->nfg", E[a], E)            # N,16,16
    F[a] = (s.max(-1).values >= 0.92).float().mean(-1)  # A'nın karelerinin B'de neredeyse aynısı olan oranı
F = torch.maximum(F, F.T).cpu().numpy(); np.fill_diagonal(F, 0)
np.save(os.path.join(RESULTS, "dino_frame_match.npy"), F)
S = np.load(os.path.join(RESULTS, "dino_video_sim.npy")); M = np.load(os.path.join(RESULTS, "dhash_match.npy"))
iu = np.triu_indices(N, 1)
P = pd.DataFrame(dict(a=iu[0], b=iu[1], fmatch=F[iu], vsim=S[iu], dhash=M[iu]))
P = P[(P.fmatch > 0) | (P.vsim > .85)]
P["cross"] = df.label.values[P.a] != df.label.values[P.b]
for t in (.25, .5, .75, 1):
    q = P[P.fmatch >= t]; print(f"frame-match>={t}: {len(q)} çift ({q.cross.sum()} farklı sınıf)")
def tile(i):
    x = np.load(os.path.join(CACHE, "clean_pairs", df.id[i] + ".npy"))[[2, 14, 26]]
    return np.hstack([cv2.resize(f, (112, 112)) for f in x])
def sheet(sub, name):
    rows = [np.hstack([tile(r.a), np.full((112, 8, 3), 255, np.uint8), tile(r.b)]) for r in sub.itertuples()]
    cv2.imwrite(os.path.join(RESULTS, name), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print(name); print(sub.assign(A=df.rel.values[sub.a], B=df.rel.values[sub.b])[["fmatch","vsim","dhash","A","B"]].to_string())
sheet(P[(P.fmatch >= .25) & (P.fmatch < .5)].sample(8, random_state=0), "dup_check_fm025.jpg")
sheet(P[(P.fmatch < .25) & (P.vsim >= .93)].sample(8, random_state=0), "dup_check_vsim093.jpg")
sheet(P[P.cross].sort_values("fmatch", ascending=False).head(8), "dup_check_cross.jpg")
P.to_csv(os.path.join(RESULTS, "similar_pairs.csv"), index=False)
