"""Kısayol (shortcut) analizi: sadece video meta verisiyle fake/real ayrılabiliyor mu?"""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
df = pd.read_csv("analysis/video_metadata.csv")
y = (df.label == "real").astype(int).values
df["aspect"] = df.width / df.height
sets = {"aspect only": ["aspect"], "width+height": ["width", "height"], "fps+duration": ["fps", "duration"],
        "all metadata": ["width", "height", "aspect", "fps", "duration", "frames", "size_mb", "bitrate_kbps"]}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for name, cols in sets.items():
    p = cross_val_predict(RandomForestClassifier(300, random_state=42), df[cols].values, y, cv=cv, method="predict_proba")[:, 1]
    print(f"{name:15s} acc={accuracy_score(y, p > .5):.4f} f1={f1_score(y, p > .5):.4f} auc={roc_auc_score(y, p):.4f}")
print("\nportrait (h>w) oranı:", df.assign(portrait=df.height > df.width).groupby("label").portrait.mean().round(3).to_dict())
