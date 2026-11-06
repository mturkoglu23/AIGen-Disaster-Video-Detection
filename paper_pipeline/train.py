"""
Ortak eğitim/değerlendirme çerçevesi (tüm modeller aynı veri, aynı bölmeler, aynı metrikler).

Örnek:
  python train.py --model orig --data raw_hybrid --split random --recipe orig --tag A_orig_raw_random
  python train.py --model tristream --data clean_pairs --split group --recipe proposed --tag P_tristream --robust
"""
import os
import sys
import json
import time
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, balanced_accuracy_score, matthews_corrcoef)

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, RESULTS
from models import build_model

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--data", required=True, choices=["raw_hybrid", "clean_hybrid", "clean_pairs", "harm288_pairs"])
p.add_argument("--split", default="group", choices=["group", "random", "lodo"])
p.add_argument("--recipe", default="proposed", choices=["orig", "fixed", "proposed", "generic"])
p.add_argument("--tag", required=True)
p.add_argument("--folds", default="all")
p.add_argument("--robust", action="store_true")
p.add_argument("--epochs", type=int, default=0)
p.add_argument("--bs", type=int, default=0)
p.add_argument("--streams", default="sem,for,tmp")
p.add_argument("--stream_drop", type=float, default=0.2)
p.add_argument("--aux_w", type=float, default=0.3)
p.add_argument("--no_jpeg_aug", action="store_true")
p.add_argument("--sem_backbone", default="vit_base_patch14_dinov2.lvd142m")
p.add_argument("--sem_unfreeze_last", type=int, default=0)
p.add_argument("--sem2_backbone", default="vit_base_patch14_dinov2.lvd142m")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--adv_w", type=float, default=0.0, help="koşullu meta veri adversaryal kaybı ağırlığı (0=kapalı)")
args = p.parse_args()

OUT = os.path.join(RESULTS, "runs", args.tag)
os.makedirs(OUT, exist_ok=True)
json.dump(vars(args), open(os.path.join(OUT, "args.json"), "w"), indent=2)
dev = "cuda"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(os.path.join(OUT, "log.txt"), "a", encoding="utf-8") as f:
        f.write(s + "\n")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------- veri
df = pd.read_csv(os.path.join(RESULTS, "splits.csv"))
t0 = time.time()
X = torch.from_numpy(np.stack([np.load(os.path.join(CACHE, args.data, i + ".npy")) for i in df.id]))
Y = torch.tensor(df.y.values, dtype=torch.float32)
# adversaryal başlık hedefleri: en-boy sınıfı (dikey / kare / yatay) ve kısa kenar çözünürlük sınıfı (<=360/<=480/<=720/>720)
_meta = pd.read_csv(os.path.join(os.path.dirname(RESULTS), "..", "analysis", "video_metadata.csv"))
_m = df.assign(file=df.rel.str.split("/").str[-1]).merge(_meta, on=["label", "disaster", "file"], how="left")
assert len(_m) == len(df) and _m.width.notna().all()
_asp = _m.width.values / _m.height.values
M = torch.tensor(np.stack([np.select([_asp < 0.9, _asp <= 1.1], [0, 1], 2),
                           np.digitize(np.minimum(_m.width.values, _m.height.values), [361, 481, 721])], 1), dtype=torch.long)
log(f"veri {args.data}: {tuple(X.shape)} {X.numel() / 2**30:.1f} GB, {time.time() - t0:.0f}s")

if args.split == "group":
    fold_of = df.fold.values
    fold_ids = list(range(5))
elif args.split == "random":
    fold_of = np.zeros(len(df), int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(df, df.y)):
        fold_of[te] = k
    fold_ids = list(range(5))
else:  # leave-one-disaster-out
    dis = sorted(df.disaster.unique())
    fold_of = df.disaster.map({d: i for i, d in enumerate(dis)}).values
    fold_ids = list(range(len(dis)))
if args.folds != "all":
    fold_ids = [int(f) for f in args.folds.split(",")]


def make_val(tr_idx, k):
    if args.split == "random":
        tr, va = train_test_split(tr_idx, test_size=0.15, stratify=df.y.values[tr_idx], random_state=42 + k)
        return tr, va
    sg = StratifiedGroupKFold(7, shuffle=True, random_state=42 + k)
    a, b = next(sg.split(tr_idx, df.y.values[tr_idx], df.group.values[tr_idx]))
    return tr_idx[a], tr_idx[b]


# ----------------------------------------------------------------------------- augment / bozulma
def jpeg_gpu(x, q):
    """x: (N,H,W,3) uint8 cuda -> JPEG kalite q ile yeniden sıkıştırılmış"""
    from torchvision.io import encode_jpeg, decode_jpeg
    imgs = list(x.permute(0, 3, 1, 2).contiguous())
    try:
        enc = encode_jpeg(imgs, quality=int(q))
        dec = decode_jpeg(enc, device="cuda")
    except Exception:
        enc = encode_jpeg([i.cpu() for i in imgs], quality=int(q))
        dec = [d.to("cuda") for d in decode_jpeg(enc)]
    return torch.stack(dec).permute(0, 2, 3, 1)


def augment(xb):
    """Klip-tutarlı yatay çevirme + p=0.3 ile JPEG(q 60-95) yeniden sıkıştırma (sosyal medya simülasyonu)."""
    b = xb.shape[0]
    flip = torch.rand(b, device=xb.device) < 0.5
    xb = torch.where(flip.view(b, 1, 1, 1, 1), xb.flip(3), xb)
    if not args.no_jpeg_aug:
        for i in range(b):
            if random.random() < 0.3:
                xb[i] = jpeg_gpu(xb[i], random.randint(60, 95))
    return xb


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
        elif kind.startswith("noise"):
            z = z + torch.randn_like(z) * float(kind[5:])
        out = z.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1)
    return out.view(b, t, h, w, c)


ROBUST = ["jpeg70", "jpeg50", "jpeg30", "down0.5", "down0.25", "blur1.0", "noise5"]


# ----------------------------------------------------------------------------- loss / metrikler
def focal_bce(logit, y, gamma=2.0):
    p = torch.sigmoid(logit)
    pt = torch.where(y > 0.5, p, 1 - p)
    return ((1 - pt) ** gamma * F.binary_cross_entropy_with_logits(logit, y, reduction="none")).mean()


def metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                auc=roc_auc_score(y, prob) if len(set(y)) > 1 else float("nan"),
                ap=average_precision_score(y, prob) if len(set(y)) > 1 else float("nan"),
                bal_acc=balanced_accuracy_score(y, pred), mcc=matthews_corrcoef(y, pred))


@torch.no_grad()
def predict(model, idx, bs, kind="none"):
    model.eval()
    probs, gates = [], []
    g = torch.Generator(device="cuda").manual_seed(0)
    for s in range(0, len(idx), bs):
        xb = X[idx[s:s + bs]].to(dev, non_blocking=True)
        if kind.startswith("noise"):
            torch.manual_seed(s)
        xb = perturb(xb, kind)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(xb)
        probs.append(torch.sigmoid(out["logit"].float()).cpu())
        if "gates" in out:
            gates.append(out["gates"].float().mean(1).cpu())
    return torch.cat(probs).numpy(), (torch.cat(gates).numpy() if gates else None)


# ----------------------------------------------------------------------------- eğitim döngüsü
def run_epochs(model, opt, sched, tr, va, epochs, bs, loss_fn, patience, plateau=None, l2=0.0, hist=None, phase=""):
    scaler = torch.amp.GradScaler()
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        perm = perm[: len(perm) // bs * bs]
        tl, t0 = 0.0, time.time()
        for s in range(0, len(perm), bs):
            bi = perm[s:s + bs]
            xb = X[bi].to(dev, non_blocking=True)
            yb = Y[bi].to(dev)
            if args.recipe != "orig":
                xb = augment(xb)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(xb)
                loss = loss_fn(out["logit"].float(), yb)
                if "aux" in out and args.aux_w > 0:
                    loss = loss + args.aux_w * torch.stack(
                        [F.binary_cross_entropy_with_logits(a.float(), yb) for a in out["aux"].values()]).mean()
                if getattr(model, "adv", None) is not None:
                    prog = (ep * len(perm) + s) / (epochs * len(perm))
                    lam = 2.0 / (1.0 + np.exp(-10 * prog)) - 1.0                      # DANN λ takvimi
                    mb = M[bi].to(dev)
                    loss = loss + args.adv_w * torch.stack(
                        [F.cross_entropy(o, mb[:, j]) for j, o in enumerate(model.adv(out["feat"], yb, lam))]).mean()
                if l2 > 0:
                    loss = loss + l2 * sum((w.float() ** 2).sum() for w in model.l2_params)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 5.0)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            tl += loss.item() * len(bi)
        vp, _ = predict(model, va, bs * 2)
        vy = Y[va].numpy()
        vauc = roc_auc_score(vy, vp)
        vloss = F.binary_cross_entropy(torch.tensor(vp).clamp(1e-6, 1 - 1e-6), torch.tensor(vy)).item()
        score = vauc - 1e-3 * vloss                          # AUC eşitliğinde düşük kayıp tercih edilir
        if plateau is not None:
            plateau.step(vauc)
        rec = dict(phase=phase, epoch=ep + 1, train_loss=tl / len(perm), val_auc=vauc, val_bce=vloss,
                   val_acc=accuracy_score(vy, vp >= .5), lr=opt.param_groups[0]["lr"], sec=time.time() - t0)
        hist.append(rec)
        log(f"  [{phase}] ep{ep + 1:02d} loss={rec['train_loss']:.4f} val_auc={vauc:.4f} val_acc={rec['val_acc']:.4f} "
            f"val_bce={vloss:.4f} ({rec['sec']:.0f}s)")
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log("  early stop")
                break
    model.load_state_dict(best_state)
    return model


def cosine_with_warmup(opt, total, warm):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, total)))))


def train_fold(k, tr, va):
    seed_all(args.seed + k)
    kw = {}
    if args.model == "tristream":
        kw = dict(streams=tuple(args.streams.split(",")), stream_drop=args.stream_drop,
                  sem_backbone=args.sem_backbone, sem_unfreeze_last=args.sem_unfreeze_last,
                  sem2_backbone=args.sem2_backbone)
    if args.model == "tristream2":
        kw = dict(streams=tuple(args.streams.split(",")), stream_drop=args.stream_drop, sem_backbone=args.sem_backbone)
    model = build_model(args.model, **kw).to(dev)
    model.adv = None
    if args.adv_w > 0:
        from models import MetaAdversary
        model.adv = MetaAdversary().to(dev)
    hist = []
    if args.recipe == "orig":
        # Orijinal Keras protokolü: Adam, focal(γ=2), bs=4, 12+23 epoch, ES(6) + ReduceLROnPlateau(0.5, 3)
        bs = args.bs or 4
        model.set_backbone_trainable("none")
        opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=1e-4)
        pl = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=3, min_lr=1e-7)
        model = run_epochs(model, opt, None, tr, va, 12, bs, focal_bce, 6, pl, l2=1e-4, hist=hist, phase="P1")
        model.set_backbone_trainable("keras_last30")
        opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=5e-6)
        pl = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=3, min_lr=1e-7)
        model = run_epochs(model, opt, None, tr, va, 23, bs, focal_bce, 6, pl, l2=1e-4, hist=hist, phase="P2")
    elif args.recipe == "fixed":
        # Düzeltilmiş strateji: AdamW + ağırlık azaltma, warmup+cosine, ayrık öğrenme oranları, augmentation,
        # tüm layer4 + layer3 son blokların ince ayarı, grup-farkında doğrulama ile model seçimi
        bs = args.bs or 8
        model.set_backbone_trainable("none")
        head = [q for q in model.parameters() if q.requires_grad]
        e1 = 6
        opt = torch.optim.AdamW(head, lr=3e-4, weight_decay=1e-2)
        sch = cosine_with_warmup(opt, e1 * (len(tr) // bs), len(tr) // bs)
        model = run_epochs(model, opt, sch, tr, va, e1, bs, focal_bce, 6, hist=hist, phase="P1")
        model.set_backbone_trainable("keras_last30")
        for n_, q in model.backbone.layer4.named_parameters():
            if "bn" not in n_ and "downsample.1" not in n_:
                q.requires_grad = True
        bb = [q for q in model.backbone.parameters() if q.requires_grad]
        e2 = args.epochs or 20
        opt = torch.optim.AdamW([{"params": bb, "lr": 2e-5}, {"params": head, "lr": 1e-4}], weight_decay=1e-2)
        sch = cosine_with_warmup(opt, e2 * (len(tr) // bs), len(tr) // bs)
        model = run_epochs(model, opt, sch, tr, va, e2, bs, focal_bce, 7, hist=hist, phase="P2")
    elif args.recipe == "proposed":
        bs = args.bs or 8
        cnn, sem, rest = [], [], []
        for n_, q in model.named_parameters():
            if not q.requires_grad:
                continue
            (cnn if n_.startswith(("for_cnn", "tmp_cnn")) else sem if n_.startswith("sem.") else rest).append(q)
        groups = [{"params": rest, "lr": 3e-4}]
        if cnn:
            groups.append({"params": cnn, "lr": 1e-4})
        if sem:
            groups.append({"params": sem, "lr": 1e-5})
        opt = torch.optim.AdamW(groups, weight_decay=0.05)
        ep = args.epochs or 25
        sch = cosine_with_warmup(opt, ep * (len(tr) // bs), len(tr) // bs)
        bce = lambda l, y: F.binary_cross_entropy_with_logits(l, y * 0.95 + 0.025)
        model = run_epochs(model, opt, sch, tr, va, ep, bs, bce, 7, hist=hist, phase="E2E")
    else:  # generic (baselines)
        bs = args.bs or 8
        opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=5e-5, weight_decay=0.05)
        ep = args.epochs or 15
        sch = cosine_with_warmup(opt, ep * (len(tr) // bs), len(tr) // bs)
        bce = lambda l, y: F.binary_cross_entropy_with_logits(l, y * 0.95 + 0.025)
        model = run_epochs(model, opt, sch, tr, va, ep, bs, bce, 5, hist=hist, phase="E2E")
    pd.DataFrame(hist).to_csv(os.path.join(OUT, f"fold{k}_history.csv"), index=False)
    return model, bs


all_rows, fold_metrics = [], []
n_params = None
for k in fold_ids:
    te = np.where(fold_of == k)[0]
    tr_all = np.where(fold_of != k)[0]
    tr, va = make_val(tr_all, k)
    log(f"\n===== {args.tag} | fold {k} | train {len(tr)} val {len(va)} test {len(te)} "
        f"| test fake/real {int((Y[te] == 0).sum())}/{int((Y[te] == 1).sum())}")
    t0 = time.time()
    model, bs = train_fold(k, tr, va)
    train_min = (time.time() - t0) / 60
    n_params = sum(q.numel() for q in model.parameters())
    vp, _ = predict(model, va, bs * 2)
    vy = Y[va].numpy()
    grid = np.arange(0.05, 0.951, 0.01)
    thr = float(grid[np.argmax([f1_score(vy, vp >= t, zero_division=0) for t in grid])])
    t1 = time.time()
    tp, gates = predict(model, te, bs * 2)
    infer_ms = (time.time() - t1) / len(te) * 1000
    ty = Y[te].numpy()
    m = metrics(ty, tp)
    mo = metrics(ty, tp, thr)
    log(f"  TEST fold{k}: " + " ".join(f"{a}={b:.4f}" for a, b in m.items()) + f" | thr*={thr:.2f} acc*={mo['accuracy']:.4f}"
        f" | {train_min:.1f} dk, {infer_ms:.1f} ms/video")
    rows = pd.DataFrame(dict(id=df.id.values[te], fold=k, y=ty, prob=tp, thr=thr))
    if gates is not None:
        for j, s in enumerate(args.streams.split(",")):
            rows[f"gate_{s}"] = gates[:, j]
    if args.robust:
        for kind in ROBUST:
            rp, _ = predict(model, te, bs * 2, kind)
            rows[f"prob_{kind}"] = rp
            log(f"    robust {kind}: acc={accuracy_score(ty, rp >= .5):.4f} auc={roc_auc_score(ty, rp):.4f}")
    all_rows.append(rows)
    fold_metrics.append(dict(fold=k, thr=thr, train_min=train_min, infer_ms=infer_ms,
                             **m, **{f"opt_{a}": b for a, b in mo.items()}))
    if args.tag.startswith(("P_", "D_", "V_")):
        frozen = {n_ for n_, q in model.named_parameters() if not q.requires_grad}   # dondurulmuş omurgalar kaydedilmez
        torch.save({n_: v for n_, v in model.state_dict().items() if n_ not in frozen}, os.path.join(OUT, f"fold{k}.pt"))
    pd.concat(all_rows).to_csv(os.path.join(OUT, "preds.csv"), index=False)
    pd.DataFrame(fold_metrics).to_csv(os.path.join(OUT, "fold_metrics.csv"), index=False)
    del model
    torch.cuda.empty_cache()

fm = pd.DataFrame(fold_metrics)
summ = fm.drop(columns="fold").agg(["mean", "std"]).T
summ.to_csv(os.path.join(OUT, "summary.csv"))
log(f"\n##### {args.tag} ÖZET (params={n_params / 1e6:.1f}M)")
log(summ.to_string(float_format=lambda v: f"{v:.4f}"))
