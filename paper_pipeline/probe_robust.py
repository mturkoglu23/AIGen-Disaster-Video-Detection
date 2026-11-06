"""
Lineer probe baseline'ları için bozulma dayanıklılığı (train.py ile aynı ROBUST listesi ve bozulma tanımları).
Probe temiz gömmelerle eğitilir (probe_baselines.py ile aynı), test videoları bozulmuş karelerden gömülür.
Sonuç: Bl_dinov2_probe / Bl_clip_univfd_probe preds.csv dosyalarına prob_<bozulma> sütunları eklenir.
"""
import os, sys, numpy as np, pandas as pd, torch, timm, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, RESULTS

ROBUST = ["jpeg70", "jpeg50", "jpeg30", "down0.5", "down0.25", "blur1.0", "noise5"]
dev = "cuda"
df = pd.read_csv(os.path.join(RESULTS, "splits.csv"))


def jpeg_gpu(x, q):
    from torchvision.io import encode_jpeg, decode_jpeg
    imgs = list(x.permute(0, 3, 1, 2).contiguous())
    try:
        dec = decode_jpeg(encode_jpeg(imgs, quality=int(q)), device="cuda")
    except Exception:
        dec = [d.to("cuda") for d in decode_jpeg(encode_jpeg([i.cpu() for i in imgs], quality=int(q)))]
    return torch.stack(dec).permute(0, 2, 3, 1)


def perturb(flat, kind):
    """flat: (T,H,W,3) uint8 cuda — train.py perturb() ile aynı"""
    if kind == "none":
        return flat
    h, w = flat.shape[1:3]
    if kind.startswith("jpeg"):
        return jpeg_gpu(flat, int(kind[4:]))
    z = flat.permute(0, 3, 1, 2).float()
    if kind.startswith("down"):
        s = float(kind[4:])
        z = F.interpolate(F.interpolate(z, scale_factor=s, mode="bilinear", antialias=True), size=(h, w), mode="bilinear")
    elif kind.startswith("blur"):
        from torchvision.transforms.functional import gaussian_blur
        z = gaussian_blur(z, 5, float(kind[4:]))
    elif kind.startswith("noise"):
        z = z + torch.randn_like(z) * float(kind[5:])
    return z.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1)


def embed_all(name, kinds):
    model = timm.create_model(name, pretrained=True, num_classes=0, img_size=224).to(dev).eval().half()
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1).half()
    std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1).half()
    E = {k: [] for k in kinds}
    with torch.no_grad():
        for i, vid in enumerate(df.id):
            x0 = torch.from_numpy(np.load(os.path.join(CACHE, "clean_pairs", vid + ".npy"))[::2]).to(dev)
            for k in kinds:
                torch.manual_seed(i)
                x = perturb(x0, k).permute(0, 3, 1, 2).half() / 255
                e = F.normalize(model((x - mean) / std).float(), dim=-1)
                E[k].append(e.mean(0).cpu().numpy())
    del model; torch.cuda.empty_cache()
    return {k: np.stack(v) for k, v in E.items()}


all_ids = pd.read_csv(os.path.join(CACHE, "videos.csv")).id.tolist()
clean = {
    "Bl_dinov2_probe": np.load(os.path.join(RESULTS, "dino_frame_emb.npy")).astype(np.float32).mean(1)[[all_ids.index(i) for i in df.id]],
    "Bl_clip_univfd_probe": np.load(os.path.join(RESULTS, "clip_l14_emb.npy")),
}
backbone = {"Bl_dinov2_probe": "vit_base_patch14_dinov2.lvd142m", "Bl_clip_univfd_probe": "vit_large_patch14_clip_224.openai"}

for run, Xc in clean.items():
    cache = os.path.join(RESULTS, f"robust_emb_{run}.npz")
    if os.path.exists(cache):
        Ep = dict(np.load(cache))
    else:
        Ep = embed_all(backbone[run], ["none"] + ROBUST)
        np.savez(cache, **Ep)
    print(run, "temiz gömme tutarlılığı (max |fark|):", float(np.abs(Ep["none"] - Xc).max()))
    out = {k: np.zeros(len(df)) for k in ROBUST}
    for k in range(5):
        te, tr = df.fold.values == k, df.fold.values != k
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)).fit(Xc[tr], df.y.values[tr])
        for kind in ROBUST:
            out[kind][te] = clf.predict_proba(Ep[kind][te])[:, 1]
    path = os.path.join(RESULTS, "runs", run, "preds.csv")
    pr = pd.read_csv(path)
    pr = pr[[c for c in pr.columns if not c.startswith("prob_")]]
    add = pd.DataFrame({"id": df.id.values, **{f"prob_{k}": v for k, v in out.items()}})
    pr = pr.merge(add, on="id", how="left")
    pr.to_csv(path, index=False)
    print(run, " ".join(f"{k}={accuracy_score(pr.y, pr[f'prob_{k}'] >= .5):.4f}/{roc_auc_score(pr.y, pr[f'prob_{k}']):.4f}" for k in ROBUST))
