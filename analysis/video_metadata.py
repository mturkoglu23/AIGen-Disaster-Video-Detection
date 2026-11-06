"""Veri kümesindeki tüm videolar için meta veri (çözünürlük, fps, süre, codec, boyut) çıkarır."""
import os, sys, cv2, pandas as pd
ROOT = os.path.join(os.environ.get("DISASTER_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "class_dataset")
rows = []
for label in ("fake", "real"):
    for disaster in sorted(os.listdir(os.path.join(ROOT, label))):
        d = os.path.join(ROOT, label, disaster)
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            cap = cv2.VideoCapture(p)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="ignore")
            cap.release()
            rows.append(dict(label=label, disaster=disaster.split("_")[0], file=f, width=w, height=h,
                             fps=round(fps, 2), frames=n, duration=round(n / fps, 2) if fps else 0,
                             size_mb=round(os.path.getsize(p) / 2**20, 2), codec=fourcc,
                             bitrate_kbps=round(os.path.getsize(p) * 8 / 1000 / (n / fps), 1) if fps and n else 0))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(os.path.dirname(__file__), "video_metadata.csv"), index=False)
print(df.groupby("label")[["width","height","fps","duration","frames","size_mb","bitrate_kbps"]].describe().T.round(2).to_string())
for c in ["codec", "fps"]:
    print("\n", pd.crosstab(df[c], df.label).sort_values("fake", ascending=False).head(12))
df["res"] = df.width.astype(str) + "x" + df.height.astype(str)
print("\n", pd.crosstab(df.res, df.label).assign(t=lambda x: x.sum(1)).sort_values("t", ascending=False).head(15))
print("\n", pd.crosstab(df.disaster, df.label, margins=True))
