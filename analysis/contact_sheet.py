"""Her sınıftan rastgele videoların orta karesini (orijinal en-boy oranıyla) bir pafta halinde kaydeder."""
import os, cv2, random, numpy as np, pandas as pd
df = pd.read_csv("analysis/video_metadata.csv"); random.seed(1)
for label in ("fake", "real"):
    sub = df[df.label == label].sample(24, random_state=3)
    tiles = []
    for _, r in sub.iterrows():
        p = os.path.join("class_dataset", label, [d for d in os.listdir(os.path.join("class_dataset", label)) if d.startswith(r.disaster)][0], r.file)
        cap = cv2.VideoCapture(p); cap.set(cv2.CAP_PROP_POS_FRAMES, r.frames // 2); ok, fr = cap.read(); cap.release()
        if not ok: fr = np.zeros((10, 10, 3), np.uint8)
        s = 300 / max(fr.shape[:2]); fr = cv2.resize(fr, None, fx=s, fy=s)
        t = np.full((300, 300, 3), 128, np.uint8); y0, x0 = (300 - fr.shape[0]) // 2, (300 - fr.shape[1]) // 2
        t[y0:y0 + fr.shape[0], x0:x0 + fr.shape[1]] = fr; tiles.append(t)
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, 24, 6)]
    cv2.imwrite(f"analysis/sheet_{label}.jpg", np.vstack(rows))
