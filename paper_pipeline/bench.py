"""
Hesaplama maliyeti karşılaştırması (aynı GPU, fp16 autocast, rastgele uint8 girdi):
  parametre (toplam / eğitilebilir), GFLOPs (video başına), gecikme (batch=1, ms/video),
  verim (batch=8, video/s), tepe GPU belleği.  python bench.py
"""
import os, sys, time, json
import numpy as np, pandas as pd, torch, timm
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
from models import build_model, to_float

dev = "cuda"
torch.backends.cudnn.benchmark = True
CLIP = "vit_large_patch14_clip_224.openai"


class ClipProbe(torch.nn.Module):
    """UnivFD tarzı: dondurulmuş CLIP ViT-L/14, 16 kare gömme ortalaması + lineer katman"""
    def __init__(self):
        super().__init__()
        self.m = timm.create_model(CLIP, pretrained=True, num_classes=0)
        for q in self.m.parameters():
            q.requires_grad = False
        c = self.m.pretrained_cfg
        self.register_buffer("mean", torch.tensor(c["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(c["std"]).view(1, 3, 1, 1))
        self.fc = torch.nn.Linear(self.m.num_features, 1)

    def forward(self, x):
        x = to_float(x)[:, 0::2]
        b, t = x.shape[:2]
        e = self.m((x.flatten(0, 1) - self.mean) / self.std).view(b, t, -1).mean(1)
        return {"logit": self.fc(e.float()).squeeze(1)}


def tri(streams, bb):
    return lambda: build_model("tristream", streams=streams, sem_backbone=bb)


MODELS = [  # (ad, kurucu, girdi kare sayısı)
    ("Orig (ResNet50+SE+BiLSTM+MHA)", lambda: build_model("orig"), 16),
    ("Xception (frame)", lambda: build_model("frame_legacy_xception"), 32),
    ("EfficientNet-B0 (frame)", lambda: build_model("frame_efficientnet_b0"), 32),
    ("R(2+1)D-18", lambda: build_model("r2plus1d"), 32),
    ("VideoMAE-B", lambda: build_model("videomae"), 32),
    ("CLIP probe (UnivFD)", ClipProbe, 32),
    ("TriStream V1 (DINOv2)", tri(("sem", "for", "tmp"), "vit_base_patch14_dinov2.lvd142m"), 32),
    ("TriStream CLIP", tri(("sem", "for", "tmp"), CLIP), 32),
    ("CLIP semantic only", tri(("sem",), CLIP), 32),
    ("Proposed: CLIP sem + temporal", tri(("sem", "tmp"), CLIP), 32),
]


def sync():
    torch.cuda.synchronize()


rows = []
for name, ctor, T in MODELS:
    m = ctor().to(dev).eval()
    tot = sum(q.numel() for q in m.parameters())
    trn = sum(q.numel() for q in m.parameters() if q.requires_grad)
    x1 = torch.randint(0, 256, (1, T, 224, 224, 3), dtype=torch.uint8, device=dev)
    x8 = x1.repeat(8, 1, 1, 1, 1)
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            m(x1)
        gflops = fc.get_total_flops() / 2e9          # MAC*2 -> FLOPs; /2 => GMACs ~ "GFLOPs" literatür geleneği
        with torch.autocast("cuda", dtype=torch.float16):
            for _ in range(10):
                m(x1)
            sync(); torch.cuda.reset_peak_memory_stats()
            lat = []
            for _ in range(50):
                t0 = time.perf_counter(); m(x1); sync(); lat.append((time.perf_counter() - t0) * 1000)
            mem = torch.cuda.max_memory_allocated() / 2**20
            for _ in range(3):
                m(x8)
            sync(); t0 = time.perf_counter()
            for _ in range(10):
                m(x8)
            sync(); thr = 80 / (time.perf_counter() - t0)
    r = dict(model=name, params_M=tot / 1e6, trainable_M=trn / 1e6, GMACs=gflops,
             latency_ms=float(np.median(lat)), latency_p90=float(np.percentile(lat, 90)),
             throughput_vps=thr, peak_mem_MB=mem)
    rows.append(r)
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, ensure_ascii=False), flush=True)
    del m, x1, x8
    torch.cuda.empty_cache()

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RESULTS, "bench.csv"), index=False)
print(out.round(2).to_string(index=False))
