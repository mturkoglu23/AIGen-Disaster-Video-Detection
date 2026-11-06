"""
Kopya analizi (3 düzey):
  1) birebir aynı dosya (MD5),
  2) piksel düzeyinde aynı/çok benzer video (kare dHash eşleşmesi; yeniden kodlanmış, kırpılmış,
     farklı çözünürlükteki kopyaları yakalar),
  3) anlamsal benzerlik (DINOv2 video gömmesi; aynı prompt'tan tekrar üretimler / aynı olayın
     farklı yüklemeleri). Bu düzey, grup bazlı çapraz doğrulama grupları için kullanılır.
"""
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import timm

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, RESULTS, ROOT

df = pd.read_csv(os.path.join(CACHE, "videos.csv"))
meta = {r: json.load(open(os.path.join(CACHE, "meta", r + ".json"))) for r in df.id}
out = {}

# --- 1) MD5
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

df["md5"] = [md5(p) for p in df.path]
dup_md5 = df[df.duplicated("md5", keep=False)].sort_values("md5")
out["exact_duplicate_files"] = int(df.md5.duplicated().sum())
print("Birebir aynı dosya (fazlalık):", out["exact_duplicate_files"])
print(dup_md5[["rel", "md5"]].to_string())

# dataset/ ile class_dataset/ tutarlılığı
ds_md5 = {}
for lab in ("fake", "real") if os.path.isdir(os.path.join(ROOT, "dataset")) else ():
    d = os.path.join(ROOT, "dataset", lab)
    for f in os.listdir(d):
        ds_md5[md5(os.path.join(d, f))] = f"{lab}/{f}"
cls_set = set(df.md5)
out["dataset_only"] = [v for k, v in ds_md5.items() if k not in cls_set]
out["class_dataset_only"] = df[~df.md5.isin(ds_md5)].rel.tolist()
print("dataset/'da olup class_dataset'te olmayan:", out["dataset_only"])
print("class_dataset'te olup dataset/'da olmayan:", out["class_dataset_only"])

# --- 2) dHash kare eşleşmesi
H = np.array([meta[i]["dhash_clean"] for i in df.id], dtype=np.uint64)          # (N,16)
Hr = np.array([meta[i]["dhash_raw"] for i in df.id], dtype=np.uint64)


def popcount64(x):
    x = x.astype(np.uint64)
    c = np.zeros(x.shape, np.uint8)
    for s in range(0, 64, 8):
        c += np.unpackbits(((x >> np.uint64(s)) & np.uint64(255)).astype(np.uint8)[..., None], axis=-1).sum(-1).astype(np.uint8)
    return c


def frame_match_matrix(H, thr=6):
    n = len(H)
    M = np.zeros((n, n), np.float32)
    for a in range(n):
        d = popcount64(H[a][None, :, None] ^ H[:, None, :])                         # (N,16,16)
        valid = H[a] != 0                                                           # siyah kareleri yok say
        hit = (d.min(-1) <= thr) & valid[None]
        M[a] = hit.sum(-1) / max(valid.sum(), 1)
    np.fill_diagonal(M, 0)
    return np.maximum(M, M.T)

Mc = frame_match_matrix(H)
Mr = frame_match_matrix(Hr)
M = np.maximum(Mc, Mr)
np.save(os.path.join(RESULTS, "dhash_match.npy"), M)

# --- 3) DINOv2 gömmeleri
dev = "cuda"
model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).to(dev).eval().half()
cfg = timm.data.resolve_data_config({}, model=model)
mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1).half()
std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1).half()
E = []
with torch.no_grad():
    for vid in df.id:
        x = np.load(os.path.join(CACHE, "clean_pairs", vid + ".npy"))[::2]
        x = torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).half() / 255
        e = model((x - mean) / std).float()
        e = torch.nn.functional.normalize(e, dim=-1)
        E.append(e.cpu().numpy())
E = np.stack(E)                                                                     # (N,16,768)
np.save(os.path.join(RESULTS, "dino_frame_emb.npy"), E.astype(np.float16))
V = E.mean(1)
V /= np.linalg.norm(V, axis=1, keepdims=True)
S = V @ V.T
np.fill_diagonal(S, 0)
np.save(os.path.join(RESULTS, "dino_video_sim.npy"), S)

# en benzer çiftleri listele
iu = np.triu_indices(len(df), 1)
pairs = pd.DataFrame(dict(a=iu[0], b=iu[1], dhash=M[iu], dino=S[iu]))
pairs = pairs[(pairs.dhash >= 0.25) | (pairs.dino >= 0.85)].copy()
pairs["rel_a"] = df.rel.values[pairs.a]
pairs["rel_b"] = df.rel.values[pairs.b]
pairs["same_label"] = df.label.values[pairs.a] == df.label.values[pairs.b]
pairs.sort_values(["dhash", "dino"], ascending=False).to_csv(os.path.join(RESULTS, "similar_pairs.csv"), index=False)
print(pairs.sort_values("dhash", ascending=False).head(40).to_string())
for t in (0.25, 0.5, 0.75, 1.0):
    print(f"dHash eşleşme >= {t}: {(pairs.dhash >= t).sum()} çift, farklı sınıf: {((pairs.dhash >= t) & ~pairs.same_label).sum()}")
for t in (0.85, 0.9, 0.93, 0.95, 0.97):
    print(f"DINO cos >= {t}: {(pairs.dino >= t).sum()} çift, farklı sınıf: {((pairs.dino >= t) & ~pairs.same_label).sum()}")
json.dump(out, open(os.path.join(RESULTS, "duplicates_summary.json"), "w"), indent=2, ensure_ascii=False)
