"""
Eğitimsiz-omurga baseline'ları (aynı grup-CV bölmeleri):
  - Meta veri Random Forest (kısayol referansı; piksel görmez)
  - UnivFD tarzı CLIP ViT-L/14 lineer probe (Ojha ve ark., CVPR 2023)
  - DINOv2 ViT-B/14 lineer probe
Video skoru = 16 karenin gömme ortalaması üzerinde lojistik regresyon.
"""
import os, sys, numpy as np, pandas as pd, torch, timm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, RESULTS

df = pd.read_csv(os.path.join(RESULTS, "splits.csv"))
meta = pd.read_csv(os.path.join(os.path.dirname(RESULTS), "..", "analysis", "video_metadata.csv"))
df["file"] = df.rel.str.split("/").str[-1]
m = df.merge(meta, on=["label", "disaster", "file"], how="left")
assert m.width.notna().all() and len(m) == len(df)
m["aspect"] = m.width / m.height
META_COLS = ["width", "height", "aspect", "fps", "duration", "frames", "size_mb", "bitrate_kbps"]


def clip_feats():
    path = os.path.join(RESULTS, "clip_l14_emb.npy")
    if os.path.exists(path):
        return np.load(path)
    model = timm.create_model("vit_large_patch14_clip_224.openai", pretrained=True, num_classes=0).cuda().eval().half()
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).cuda().half()
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).cuda().half()
    E = []
    with torch.no_grad():
        for vid in df.id:
            x = torch.from_numpy(np.load(os.path.join(CACHE, "clean_pairs", vid + ".npy"))[::2]).cuda()
            x = x.permute(0, 3, 1, 2).half() / 255
            e = torch.nn.functional.normalize(model((x - mean) / std).float(), dim=-1)
            E.append(e.mean(0).cpu().numpy())
    E = np.stack(E)
    np.save(path, E)
    return E


def run(name, Xf, make):
    rows = []
    for k in range(5):
        te, tr = df.fold.values == k, df.fold.values != k
        clf = make().fit(Xf[tr], df.y.values[tr])
        rows.append(pd.DataFrame(dict(id=df.id.values[te], fold=k, y=df.y.values[te], prob=clf.predict_proba(Xf[te])[:, 1], thr=0.5)))
    out = os.path.join(RESULTS, "runs", name)
    os.makedirs(out, exist_ok=True)
    pr = pd.concat(rows)
    pr.to_csv(os.path.join(out, "preds.csv"), index=False)
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    print(f"{name:22s} acc={accuracy_score(pr.y, pr.prob >= .5):.4f} f1={f1_score(pr.y, pr.prob >= .5):.4f} auc={roc_auc_score(pr.y, pr.prob):.4f}")


run("Bl_metadata_rf", m[META_COLS].values, lambda: RandomForestClassifier(500, random_state=42, n_jobs=-1))
run("Bl_metadata_rf_aspect", m[["aspect"]].values, lambda: RandomForestClassifier(500, random_state=42, n_jobs=-1))
all_ids = pd.read_csv(os.path.join(CACHE, "videos.csv")).id.tolist()
dino = np.load(os.path.join(RESULTS, "dino_frame_emb.npy")).astype(np.float32).mean(1)[[all_ids.index(i) for i in df.id]]
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))
run("Bl_dinov2_probe", dino, lr)
run("Bl_clip_univfd_probe", clip_feats(), lr)
