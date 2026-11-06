"""Ortak yollar ve video listesi. Tüm deneyler class_dataset'i ana kaynak olarak kullanır."""
import os
import hashlib
import pandas as pd

# Repository root; override with the DISASTER_ROOT environment variable if the data live elsewhere.
ROOT = os.environ.get("DISASTER_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASS_DIR = os.path.join(ROOT, "class_dataset")
WORK = os.path.join(ROOT, "paper_pipeline")
CACHE = os.path.join(WORK, "cache")
RESULTS = os.path.join(WORK, "results")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

NUM_FRAMES = 16
IMG = 224
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def vid_id(rel_path: str) -> str:
    return hashlib.md5(rel_path.lower().encode("utf-8")).hexdigest()[:12]


def list_videos() -> pd.DataFrame:
    rows = []
    for label in ("fake", "real"):
        for disaster in sorted(os.listdir(os.path.join(CLASS_DIR, label))):
            d = os.path.join(CLASS_DIR, label, disaster)
            for f in sorted(os.listdir(d)):
                if os.path.splitext(f)[1].lower() in VIDEO_EXT:
                    rel = f"{label}/{disaster}/{f}"
                    rows.append(dict(id=vid_id(rel), rel=rel, path=os.path.join(d, f), label=label,
                                     y=int(label == "real"), disaster=disaster.split("_")[0]))
    return pd.DataFrame(rows)
