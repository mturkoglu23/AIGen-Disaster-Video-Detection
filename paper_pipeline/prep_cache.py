"""
Tek geçişli ön işleme: her video için

  raw_hybrid   (16,224,224,3)  orijinal protokol: tüm kare 224x224'e sıkıştırılır (en-boy bozulur),
                               12 uniform + 4 hareket-tabanlı kare (deterministik).
  clean_hybrid (16,224,224,3)  kısayol-kontrollü protokol, aynı indeksler.
  clean_pairs  (32,224,224,3)  kısayol-kontrollü protokol, 16 uniform konum x (t, t+1) kare çifti
                               (zamansal/residual akış için).

Kısayol-kontrollü (clean) dönüşüm:
  1) letterbox / pillarbox / bulanık dolgu bantları otomatik kırpılır,
  2) her kenardan %10 kenar payı atılır (filigran, logo, zaman damgası, altyazı bantları),
  3) en-boy oranı korunarak kısa kenar 224'e indirilir ve merkezden 224x224 kırpılır.

Ayrıca kopya tespiti için her uniform karenin 64-bit dHash'i ve video meta bilgisi kaydedilir.
"""
import os
import sys
import json
import numpy as np
import cv2
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, NUM_FRAMES, IMG, list_videos

MARGIN = 0.10
FLOW_SIZE = 112
UNIFORM_N = 12
MOTION_N = NUM_FRAMES - UNIFORM_N
OUT_DIRS = {k: os.path.join(CACHE, k) for k in ("raw_hybrid", "clean_hybrid", "clean_pairs")}
for d in OUT_DIRS.values():
    os.makedirs(d, exist_ok=True)
META_DIR = os.path.join(CACHE, "meta")
os.makedirs(META_DIR, exist_ok=True)


def detect_content_rect(gray_stack):
    """gray_stack: (T,H,W) küçük ölçekli gri kareler. Statik ve düzgün kenar bantlarını bulur."""
    t, h, w = gray_stack.shape
    g = gray_stack.astype(np.float32)
    temporal_std = g.std(0)                      # (H,W)
    mean_img = g.mean(0)

    def is_bar(line_mean, line_tstd, line_mean_vals):
        dark = line_mean_vals.mean() < 28 and line_mean_vals.std() < 12
        static_flat = line_tstd.mean() < 3.0 and line_mean_vals.std() < 10
        return dark or static_flat

    def scan(n, get):
        k = 0
        while k < int(n * 0.35) and is_bar(*get(k)):
            k += 1
        return k

    top = scan(h, lambda i: (None, temporal_std[i], mean_img[i]))
    bottom = scan(h, lambda i: (None, temporal_std[h - 1 - i], mean_img[h - 1 - i]))
    left = scan(w, lambda i: (None, temporal_std[:, i], mean_img[:, i]))
    right = scan(w, lambda i: (None, temporal_std[:, w - 1 - i], mean_img[:, w - 1 - i]))
    # bulanık dolgu (Instagram/TikTok tarzı) için: kenarlarda çok düşük gradyan enerjisi
    return top / h, bottom / h, left / w, right / w


def clean_transform(frame, rect):
    h, w = frame.shape[:2]
    t, b, l, r = rect
    y0, y1 = int(h * t), int(h * (1 - b))
    x0, x1 = int(w * l), int(w * (1 - r))
    if y1 - y0 < h * 0.3 or x1 - x0 < w * 0.3:
        y0, y1, x0, x1 = 0, h, 0, w
    f = frame[y0:y1, x0:x1]
    hh, ww = f.shape[:2]
    my, mx = int(hh * MARGIN), int(ww * MARGIN)
    f = f[my:hh - my, mx:ww - mx]
    hh, ww = f.shape[:2]
    s = IMG / min(hh, ww)
    f = cv2.resize(f, (max(IMG, round(ww * s)), max(IMG, round(hh * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    hh, ww = f.shape[:2]
    y, x = (hh - IMG) // 2, (ww - IMG) // 2
    return f[y:y + IMG, x:x + IMG]


def raw_transform(frame):
    return cv2.resize(frame, (IMG, IMG))


def dhash(img_rgb):
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hybrid_indices(n, scores):
    uni = np.linspace(0, n - 1, UNIFORM_N).round().astype(int).tolist()
    chosen = list(uni)
    min_gap = max(1, n // (NUM_FRAMES * 2))
    motion = []
    for i in np.argsort(-scores):
        if len(motion) >= MOTION_N:
            break
        if all(abs(int(i) - c) >= min_gap for c in chosen):
            motion.append(int(i))
            chosen.append(int(i))
    for i in np.linspace(0, n - 1, NUM_FRAMES * 2).round().astype(int):
        if len(chosen) >= NUM_FRAMES:
            break
        if int(i) not in chosen:
            chosen.append(int(i))
    chosen = sorted(chosen)[:NUM_FRAMES]
    while len(chosen) < NUM_FRAMES:
        chosen.append(chosen[-1])
    return chosen, motion


def process(row):
    vid, path = row["id"], row["path"]
    outs = {k: os.path.join(d, vid + ".npy") for k, d in OUT_DIRS.items()}
    meta_path = os.path.join(META_DIR, vid + ".json")
    if all(os.path.exists(p) for p in outs.values()) and os.path.exists(meta_path):
        return vid, "cached"

    # --- 1. geçiş: hareket skorları + küçük gri kareler (letterbox tespiti)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    scores, smalls = [], []
    prev = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        gs = cv2.resize(g, (FLOW_SIZE, FLOW_SIZE), interpolation=cv2.INTER_AREA)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, gs, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            scores.append(float(np.mean(np.abs(flow))))
        else:
            scores.append(0.0)
        prev = gs
        smalls.append(cv2.resize(g, (g.shape[1] // 4, g.shape[0] // 4), interpolation=cv2.INTER_AREA))
    cap.release()
    n = len(scores)
    if n < 2:
        return vid, "unreadable"
    scores = np.asarray(scores, np.float32)
    samp = np.linspace(0, n - 1, min(n, 32)).astype(int)
    rect = detect_content_rect(np.stack([smalls[i] for i in samp]))
    del smalls

    uni = np.linspace(0, n - 2, NUM_FRAMES).round().astype(int).tolist()
    pair_idx = [i for u in uni for i in (u, u + 1)]
    hyb, motion = hybrid_indices(n, scores)
    need = set(pair_idx) | set(hyb)

    # --- 2. geçiş: gereken kareleri sıralı oku
    frames = {}
    cap = cv2.VideoCapture(path)
    i, last = 0, max(need)
    while i <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if i in need:
            frames[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()

    def get(idx):
        if idx in frames:
            return frames[idx]
        k = max([j for j in frames if j <= idx], default=min(frames))
        return frames[k]

    raw_h = np.stack([raw_transform(get(j)) for j in hyb])
    clean_h = np.stack([clean_transform(get(j), rect) for j in hyb])
    clean_p = np.stack([clean_transform(get(j), rect) for j in pair_idx])
    np.save(outs["raw_hybrid"], raw_h)
    np.save(outs["clean_hybrid"], clean_h)
    np.save(outs["clean_pairs"], clean_p)

    h0, w0 = next(iter(frames.values())).shape[:2]
    meta = dict(id=vid, n_frames=n, fps=fps, width=w0, height=h0, rect=rect,
                hybrid_idx=hyb, motion_idx=motion, uniform_idx=uni,
                dhash_clean=[dhash(f) for f in clean_p[::2]],
                dhash_raw=[dhash(raw_transform(get(j))) for j in uni],
                motion_mean=float(scores.mean()), motion_std=float(scores.std()))
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return vid, "ok"


if __name__ == "__main__":
    df = list_videos()
    df.to_csv(os.path.join(CACHE, "videos.csv"), index=False)
    print(f"{len(df)} video")
    rows = df.to_dict("records")
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 16) as pool:
        for k, (vid, st) in enumerate(pool.imap_unordered(process, rows, chunksize=1), 1):
            if st != "ok" or k % 50 == 0:
                print(k, vid, st, flush=True)
    print("DONE")
