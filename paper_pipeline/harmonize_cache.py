"""Hakem M1: çözünürlük-uyumlaştırılmış test girdisi.
clean_pairs ile aynı kareler ve aynı kırpma; tek fark: kenar payı kırpımından sonra içerik kısa kenarı
TARGET pikselden büyükse önce INTER_AREA ile TARGET'a indirilir (gerçek videoların medyan etkin kısa kenarı ~288),
sonra her zamanki gibi 224'e ölçeklenir. Böylece tüm videolar aynı (düşük) kaynak çözünürlüğünden gelir.
İsteğe bağlı --crf: içerik önce H.264 ile (libx264, verilen CRF) yeniden kodlanır (ffmpeg gerekir).
python harmonize_cache.py [TARGET=288] -> cache/harm{TARGET}_pairs/
"""
import os, sys, json
import numpy as np, cv2
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, IMG, list_videos
from prep_cache import MARGIN

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 288
OUT = os.path.join(CACHE, f"harm{TARGET}_pairs")
os.makedirs(OUT, exist_ok=True)


def transform(frame, rect):
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
    if min(hh, ww) > TARGET:                                   # uyumlaştırma adımı
        s = TARGET / min(hh, ww)
        f = cv2.resize(f, (round(ww * s), round(hh * s)), interpolation=cv2.INTER_AREA)
        hh, ww = f.shape[:2]
    s = IMG / min(hh, ww)
    f = cv2.resize(f, (max(IMG, round(ww * s)), max(IMG, round(hh * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    hh, ww = f.shape[:2]
    y, x = (hh - IMG) // 2, (ww - IMG) // 2
    return f[y:y + IMG, x:x + IMG]


def process(row):
    vid, path = row["id"], row["path"]
    out = os.path.join(OUT, vid + ".npy")
    mp = os.path.join(CACHE, "meta", vid + ".json")
    if os.path.exists(out) or not os.path.exists(mp):
        return vid, "skip"
    m = json.load(open(mp))
    pair_idx = [i for u in m["uniform_idx"] for i in (u, u + 1)]
    need, frames = set(pair_idx), {}
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
        return frames[max([j for j in frames if j <= idx], default=min(frames))]

    np.save(out, np.stack([transform(get(j), m["rect"]) for j in pair_idx]))
    return vid, "ok"


if __name__ == "__main__":
    import pandas as pd
    from common import RESULTS
    keep = set(pd.read_csv(os.path.join(RESULTS, "splits.csv")).id)
    rows = [r for r in list_videos().to_dict("records") if r["id"] in keep]
    with Pool(12) as pool:
        for k, (vid, st) in enumerate(pool.imap_unordered(process, rows, chunksize=2), 1):
            if k % 200 == 0:
                print(k, flush=True)
    print("done", len(os.listdir(OUT)))
