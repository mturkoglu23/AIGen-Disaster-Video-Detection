"""
Ana model (CLIP anlamsal + zamansal) için açıklanabilirlik analizleri.
  1) Kapı ağırlıkları: temiz ve bozulmuş girdide (jpeg70, down0.25, blur1.0) video başına ortalama kapı
  2) Kol kapatma (modality knock-out): bir kolun token'ı sıfırlanınca (eğitimdeki stream-dropout ile aynı) tahmin
  3) Grad-CAM: CLIP son blok patch token'ları (16x16) ve zamansal ResNet18 layer4 (7x7) üzerinde,
     tahmin edilen sınıfın kanıtı için; ayrıca CAM enerjisinin kenar bandındaki payı (kısayol kontrolü)
Kullanım:  python explain.py --tag P_clip_semtmp_s1
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, RESULTS
from models import build_model, to_float, imnet_norm

p = argparse.ArgumentParser()
p.add_argument("--tag", default="P_clip_semtmp_s1")
p.add_argument("--bs", type=int, default=8)
args = p.parse_args()
RUN = os.path.join(RESULTS, "runs", args.tag)
A = json.load(open(os.path.join(RUN, "args.json")))
OUT = os.path.join(RESULTS, "explain", args.tag)
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
STREAMS = tuple(A["streams"].split(","))
df = pd.read_csv(os.path.join(RESULTS, "splits.csv"))
preds = pd.read_csv(os.path.join(RUN, "preds.csv")).set_index("id")


# --------------------------------------------------------------------- bozulmalar (train.py ile aynı)
def jpeg_gpu(x, q):
    from torchvision.io import encode_jpeg, decode_jpeg
    imgs = list(x.permute(0, 3, 1, 2).contiguous())
    try:
        dec = decode_jpeg(encode_jpeg(imgs, quality=int(q)), device="cuda")
    except Exception:
        dec = [d.to("cuda") for d in decode_jpeg(encode_jpeg([i.cpu() for i in imgs], quality=int(q)))]
    return torch.stack(dec).permute(0, 2, 3, 1)


def perturb(xb, kind):
    if kind == "none":
        return xb
    b, t, h, w, c = xb.shape
    flat = xb.flatten(0, 1)
    if kind.startswith("jpeg"):
        out = jpeg_gpu(flat, int(kind[4:]))
    else:
        z = flat.permute(0, 3, 1, 2).float()
        if kind.startswith("down"):
            s = float(kind[4:])
            z = F.interpolate(F.interpolate(z, scale_factor=s, mode="bilinear", antialias=True), size=(h, w), mode="bilinear")
        elif kind.startswith("blur"):
            from torchvision.transforms.functional import gaussian_blur
            z = gaussian_blur(z, 5, float(kind[4:]))
        out = z.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1)
    return out.view(b, t, h, w, c)


# --------------------------------------------------------------------- model ileri geçişi (kol kapatma + CAM için açık)
def forward(m, x, drop=None, cam=False):
    """drop: kapatılacak kol adı. cam=True ise ara aktivasyonlar ve gradyanlar için graf tutulur."""
    b = x.shape[0]
    x = to_float(x)
    f, g = x[:, 0::2], x[:, 1::2]
    t = f.shape[1]
    ff = f.flatten(0, 1)
    store = {}
    toks = {}
    if "sem" in STREAMS:
        xs = (ff - m.sem_mean) / m.sem_std
        if cam:
            xs = xs.requires_grad_(True)
        h = m.sem.blocks[-1].register_forward_hook(lambda mod, i, o: store.__setitem__("sem", o))
        inter = m.sem.get_intermediate_layers(xs, n=m.sem_n, return_prefix_tokens=True, norm=True)
        h.remove()
        feats = [c[:, 0] for _, c in inter] + [inter[-1][0].mean(1)]
        toks["sem"] = m.sem_proj(torch.cat(feats, -1).float()).view(b, t, -1)
    if "tmp" in STREAMS:
        diff = (g.flatten(0, 1) - ff) * 4
        tin = torch.cat([imnet_norm(ff), diff], 1)
        if cam:
            tin = tin.requires_grad_(True)
        h = m.tmp_cnn.layer4.register_forward_hook(lambda mod, i, o: store.__setitem__("tmp", o))
        toks["tmp"] = m.tmp_proj(m.tmp_cnn(tin)).view(b, t, -1)
        h.remove()
    S = torch.stack([toks[s] + m.stream_emb[i] for i, s in enumerate(STREAMS)], 2)
    if drop is not None:
        keep = torch.ones(1, 1, len(STREAMS), 1, device=S.device)
        keep[:, :, STREAMS.index(drop)] = 0
        S = S * keep
    w = torch.softmax(m.gate(S.flatten(2)), -1)
    z = (S * w.unsqueeze(-1)).sum(2)
    z = torch.cat([m.cls.expand(b, -1, -1), z], 1) + m.pos[:, : t + 1]
    z = m.norm(m.temporal(z))
    return m.head(z[:, 0]).squeeze(1), w, store


def load_model(k):
    kw = dict(streams=STREAMS, stream_drop=A["stream_drop"], sem_backbone=A["sem_backbone"],
              sem_unfreeze_last=A["sem_unfreeze_last"], sem2_backbone=A["sem2_backbone"])
    m = build_model("tristream", **kw)
    sd = torch.load(os.path.join(RUN, f"fold{k}.pt"), map_location="cpu")
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not unexp and all(n.startswith(("sem.", "sem2.")) for n in miss), (unexp, miss[:5])
    return m.to(dev).eval()


def load_x(ids):
    return torch.from_numpy(np.stack([np.load(os.path.join(CACHE, A["data"], i + ".npy")) for i in ids]))


def cam_maps(act, grad, vit):
    """act/grad -> (T,H,W) normalize edilmemiş, ReLU'lu CAM"""
    if vit:   # (T, 1+N, C) -> patch tokenlar
        act, grad = act[:, 1:], grad[:, 1:]
        n = int(round(act.shape[1] ** 0.5))
        wts = grad.mean(1, keepdim=True)
        c = F.relu((act * wts).sum(-1)).view(-1, n, n)
    else:     # (T, C, h, w)
        wts = grad.mean((2, 3), keepdim=True)
        c = F.relu((act * wts).sum(1))
    return c.float()


def border_share(c, frac=0.15):
    """CAM enerjisinin dış %frac bandındaki payı (T,H,W) -> skaler; düzgün dağılımda 1-(1-2f)^2"""
    c = F.interpolate(c[:, None], size=(224, 224), mode="bilinear")[:, 0]
    tot = c.sum((1, 2)) + 1e-8
    b = int(224 * frac)
    inner = c[:, b:-b, b:-b].sum((1, 2))
    return ((tot - inner) / tot).mean().item()


KINDS = ["none", "jpeg70", "down0.25", "blur1.0"]
rows, cams = [], {}
sel = None
for k in range(5):
    m = load_model(k)
    te = df[df.fold == k]
    ids = te.id.values
    X = load_x(ids)
    res = {"id": ids, "fold": k, "y": te.y.values, "disaster": te.disaster.values}
    # 1) + 2) kapı ve kol kapatma
    with torch.no_grad():
        for kind in KINDS:
            pr, gt = [], []
            for s in range(0, len(ids), args.bs):
                xb = perturb(X[s:s + args.bs].to(dev), kind)
                with torch.autocast("cuda", dtype=torch.float16):
                    lg, w, _ = forward(m, xb)
                pr.append(torch.sigmoid(lg.float()).cpu()); gt.append(w.float().mean(1).cpu())
            res[f"prob_{kind}"] = torch.cat(pr).numpy()
            gt = torch.cat(gt).numpy()
            for j, s_ in enumerate(STREAMS):
                res[f"gate_{s_}_{kind}"] = gt[:, j]
        for d in STREAMS:
            pr = []
            for s in range(0, len(ids), args.bs):
                with torch.autocast("cuda", dtype=torch.float16):
                    lg, _, _ = forward(m, X[s:s + args.bs].to(dev), drop=d)
                pr.append(torch.sigmoid(lg.float()).cpu())
            res[f"prob_drop_{d}"] = torch.cat(pr).numpy()
    # 3) Grad-CAM (her video, temiz girdi, fp32)
    bsem, btmp = [], []
    for i, vid in enumerate(ids):
        xb = X[i:i + 1].to(dev)
        lg, _, st = forward(m, xb, cam=True)
        for v in st.values():
            v.retain_grad()
        sign = 1.0 if lg.item() >= 0 else -1.0     # tahmin edilen sınıfın kanıtı (y=1 gerçek)
        m.zero_grad(set_to_none=True)
        (sign * lg).sum().backward()
        cs = cam_maps(st["sem"].detach(), st["sem"].grad, True)
        ct = cam_maps(st["tmp"].detach(), st["tmp"].grad, False)
        bsem.append(border_share(cs)); btmp.append(border_share(ct))
        cams[vid] = (cs.cpu().numpy().astype(np.float16), ct.cpu().numpy().astype(np.float16))
    res["border_sem"], res["border_tmp"] = bsem, btmp
    rows.append(pd.DataFrame(res))
    print(f"fold{k} tamam", flush=True)
    del m, X
    torch.cuda.empty_cache()

R = pd.concat(rows).reset_index(drop=True)
R = R.merge(preds[["thr"]], left_on="id", right_index=True)
R.to_csv(os.path.join(OUT, "explain.csv"), index=False)
np.savez_compressed(os.path.join(OUT, "cams.npz"), **{f"{k}_sem": v[0] for k, v in cams.items()},
                    **{f"{k}_tmp": v[1] for k, v in cams.items()})
print("kaydedildi", OUT)
