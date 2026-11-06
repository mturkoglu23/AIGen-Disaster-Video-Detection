"""
Modeller:
  OrigNet       : Kullanıcının önerdiği ResNet50 + SE + 2xBiLSTM + MHA + FFN mimarisinin PyTorch karşılığı
                  (Keras model_proposed_1.py ile katman katman aynı).
  TriStreamNet  : Önerilen üç akışlı uzamsal-adli-zamansal model (DINOv2 semantik akış, SRM residual adli akış,
                  kare-farkı hareket akışı; kapılı füzyon + zamansal transformer + derin denetim).
  R2Plus1D, VideoMAE, FrameCNN, LinearProbe : karşılaştırma modelleri.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_float(x):
    """(B,T,H,W,3) uint8 -> (B,T,3,H,W) float 0..1"""
    return x.permute(0, 1, 4, 2, 3).float().div_(255)


def imnet_norm(x):
    return (x - IMNET_MEAN.to(x.device, x.dtype)) / IMNET_STD.to(x.device, x.dtype)


# =============================================================================
# 1) Kullanıcının modeli (OrigNet)
# =============================================================================
class KerasMHA(nn.Module):
    """Keras MultiHeadAttention(num_heads=8, key_dim=64): iç boyut 8*64=512, çıkış girdi boyutuna döner."""

    def __init__(self, d, heads=8, key_dim=64, dropout=0.1):
        super().__init__()
        self.h, self.k = heads, key_dim
        self.q = nn.Linear(d, heads * key_dim)
        self.kp = nn.Linear(d, heads * key_dim)
        self.v = nn.Linear(d, heads * key_dim)
        self.o = nn.Linear(heads * key_dim, d)
        self.drop = dropout

    def forward(self, x):
        b, t, _ = x.shape
        sp = lambda z: z.view(b, t, self.h, self.k).transpose(1, 2)
        y = F.scaled_dot_product_attention(sp(self.q(x)), sp(self.kp(x)), sp(self.v(x)),
                                           dropout_p=self.drop if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(b, t, -1))


class OrigNet(nn.Module):
    def __init__(self, lstm_units=256, dropout=0.5):
        super().__init__()
        r = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        r.fc = nn.Identity()
        self.backbone = r
        self.proj = nn.Linear(2048, 512)
        self.frame_ln = nn.LayerNorm(512)
        self.se1 = nn.Linear(512, 32)
        self.se2 = nn.Linear(32, 512)
        self.drop_l1 = nn.Dropout(0.3)
        self.lstm1 = nn.LSTM(512, lstm_units, batch_first=True, bidirectional=True)
        self.ln1 = nn.LayerNorm(2 * lstm_units)
        self.drop_l2 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(2 * lstm_units, lstm_units // 2, batch_first=True, bidirectional=True)
        d = lstm_units
        self.ln2 = nn.LayerNorm(d)
        self.mha = KerasMHA(d)
        self.ln_att = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(0.1), nn.Linear(2 * d, d))
        self.ln_ffn = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout / 2), nn.Linear(128, 1))
        # Keras l2(1e-4) uygulanan katmanlar
        self.l2_params = [self.proj.weight, self.lstm1.weight_ih_l0, self.lstm1.weight_ih_l0_reverse,
                          self.head[0].weight]
        self.set_backbone_trainable("none")

    def set_backbone_trainable(self, mode):
        """none: tamamen donuk | keras_last30: conv5_block2-3 (+block1 conv3) açık, BN donuk | all"""
        for p in self.backbone.parameters():
            p.requires_grad = False
        if mode == "keras_last30":
            l4 = self.backbone.layer4
            mods = [l4[1], l4[2], l4[0].conv3]
        elif mode == "all":
            mods = [self.backbone]
        else:
            mods = []
        for m in mods:
            for name, p in m.named_parameters():
                if "bn" not in name and "downsample.1" not in name:
                    p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # BN her zaman inference modunda (Keras training=False ile aynı)
        return self

    def forward(self, x):
        b, t = x.shape[:2]
        z = imnet_norm(to_float(x).flatten(0, 1))
        f = self.backbone(z).view(b, t, -1)
        f = self.frame_ln(F.relu(self.proj(f)))
        s = torch.sigmoid(self.se2(F.relu(self.se1(f.mean(1)))))
        f = f * s[:, None]
        f = self.ln1(self.lstm1(self.drop_l1(f))[0])
        f = self.ln2(self.lstm2(self.drop_l2(f))[0])
        f = self.ln_att(f + self.mha(f))
        f = self.ln_ffn(f + self.ffn(f))
        return {"logit": self.head(f.mean(1)).squeeze(1)}


# =============================================================================
# 2) Önerilen model: TriStreamNet
# =============================================================================
class SRMResidual(nn.Module):
    """Sabit SRM yüksek geçiren filtreleri (3 çekirdek) -> gürültü kalıntısı (noise residual)."""

    def __init__(self):
        super().__init__()
        k1 = torch.tensor([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]) / 4.
        k2 = torch.tensor([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]) / 12.
        k3 = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 2.
        self.register_buffer("w", torch.stack([k1, k2, k3]).float().unsqueeze(1))

    def forward(self, x):  # x (N,3,H,W) 0..1
        g = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]) * 255
        r = F.conv2d(g, self.w.to(g.dtype), padding=2)
        return torch.clamp(r, -3, 3) / 3


class TriStreamNet(nn.Module):
    def __init__(self, d=256, streams=("sem", "for", "tmp"), sem_layers=None, stream_drop=0.2,
                 sem_backbone="vit_base_patch14_dinov2.lvd142m", sem_unfreeze_last=0,
                 sem2_backbone="vit_base_patch14_dinov2.lvd142m"):
        super().__init__()
        self.streams = streams
        self.stream_drop = stream_drop
        if "sem" in streams:
            self.sem = timm.create_model(sem_backbone, pretrained=True, num_classes=0, img_size=224)
            for p in self.sem.parameters():
                p.requires_grad = False
            if sem_unfreeze_last:
                for blk in self.sem.blocks[-sem_unfreeze_last:]:
                    for p in blk.parameters():
                        p.requires_grad = True
            cfg = self.sem.pretrained_cfg
            self.register_buffer("sem_mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
            self.register_buffer("sem_std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
            depth = len(self.sem.blocks)
            if sem_layers is None:  # derinliğin 1/3, 2/3 ve sonu (ViT-B: 3,7,11 | ViT-L: 7,15,23)
                sem_layers = (depth // 3 - 1, 2 * depth // 3 - 1)
            self.sem_n = sorted(set(sem_layers) | {depth - 1})
            ed = self.sem.embed_dim
            self.sem_proj = nn.Sequential(nn.LayerNorm(ed * (len(self.sem_n) + 1)),
                                          nn.Linear(ed * (len(self.sem_n) + 1), d), nn.GELU())
        if "sem2" in streams:
            # ikinci (tamamlayıcı) dondurulmuş temel model: ör. sem=CLIP-L, sem2=DINOv2-B
            self.sem2 = timm.create_model(sem2_backbone, pretrained=True, num_classes=0, img_size=224)
            for p in self.sem2.parameters():
                p.requires_grad = False
            cfg2 = self.sem2.pretrained_cfg
            self.register_buffer("sem2_mean", torch.tensor(cfg2["mean"]).view(1, 3, 1, 1))
            self.register_buffer("sem2_std", torch.tensor(cfg2["std"]).view(1, 3, 1, 1))
            d2 = len(self.sem2.blocks)
            self.sem2_n = sorted({d2 // 3 - 1, 2 * d2 // 3 - 1, d2 - 1})
            ed2 = self.sem2.embed_dim
            self.sem2_proj = nn.Sequential(nn.LayerNorm(ed2 * (len(self.sem2_n) + 1)),
                                           nn.Linear(ed2 * (len(self.sem2_n) + 1), d), nn.GELU())
        if "for" in streams:
            self.srm = SRMResidual()
            self.for_cnn = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
            self.for_proj = nn.Sequential(nn.LayerNorm(1280), nn.Linear(1280, d), nn.GELU())
        if "tmp" in streams:
            self.tmp_cnn = timm.create_model("resnet18", pretrained=True, num_classes=0, in_chans=6)
            self.tmp_proj = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d), nn.GELU())
        k = len(streams)
        self.stream_emb = nn.Parameter(torch.zeros(k, d))
        self.gate = nn.Sequential(nn.Linear(k * d, d), nn.GELU(), nn.Linear(d, k))
        self.pos = nn.Parameter(torch.zeros(1, 17, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, 1))
        self.aux = nn.ModuleDict({s: nn.Linear(d, 1) for s in streams})

    def sem_features(self, x, which="sem"):
        net, layers = getattr(self, which), getattr(self, which + "_n")
        x = (x - getattr(self, which + "_mean")) / getattr(self, which + "_std")
        grad = any(p.requires_grad for p in net.parameters())
        with torch.set_grad_enabled(grad and self.training):
            inter = net.get_intermediate_layers(x, n=layers, return_prefix_tokens=True, norm=True)
        feats = [c[:, 0] for _, c in inter]          # ara katman + son katman CLS
        feats.append(inter[-1][0].mean(1))           # son katman ortalama patch token
        return torch.cat(feats, -1).float()

    def forward(self, x):
        # x: (B,32,H,W,3) -> 16 konum x (t, t+1)
        b = x.shape[0]
        x = to_float(x)
        f, g = x[:, 0::2], x[:, 1::2]
        t = f.shape[1]
        ff = f.flatten(0, 1)
        toks = {}
        if "sem" in self.streams:
            toks["sem"] = self.sem_proj(self.sem_features(ff)).view(b, t, -1)
        if "sem2" in self.streams:
            toks["sem2"] = self.sem2_proj(self.sem_features(ff, "sem2")).view(b, t, -1)
        if "for" in self.streams:
            toks["for"] = self.for_proj(self.for_cnn(self.srm(ff))).view(b, t, -1)
        if "tmp" in self.streams:
            gg = g.flatten(0, 1)
            diff = (gg - ff) * 4
            tin = torch.cat([imnet_norm(ff), diff], 1)
            toks["tmp"] = self.tmp_proj(self.tmp_cnn(tin)).view(b, t, -1)
        S = torch.stack([toks[s] + self.stream_emb[i] for i, s in enumerate(self.streams)], 2)   # B,T,K,d
        if self.training and self.stream_drop > 0 and len(self.streams) > 1:
            keep = (torch.rand(b, 1, len(self.streams), 1, device=S.device) > self.stream_drop).float()
            keep[keep.sum(2, keepdim=True).expand_as(keep) == 0] = 1.0
            S = S * keep
        w = torch.softmax(self.gate(S.flatten(2)), -1)                                             # B,T,K
        z = (S * w.unsqueeze(-1)).sum(2)
        z = torch.cat([self.cls.expand(b, -1, -1), z], 1) + self.pos[:, : t + 1]
        z = self.norm(self.temporal(z))
        out = {"logit": self.head(z[:, 0]).squeeze(1), "gates": w.detach()}
        out["aux"] = {s: self.aux[s](toks[s].mean(1)).squeeze(1) for s in self.streams}
        return out


class TriStreamV2(nn.Module):
    """
    V2: zamansal bilgi ayrı bir piksel-CNN yerine dondurulmuş temel modelin özellik uzayından çıkarılır.
      sem  : t karesi -> ara katman + son katman CLS ve ortalama patch token (V1 ile aynı)
      ftmp : (t, t+1) patch token'larının farkı ve patch-bazlı kosinüs benzerliği istatistikleri
             (ara ve son katman) -> anlamsal düzeyde kareler arası tutarlılık
      for  : SRM kalıntısı -> EfficientNet-B0 (isteğe bağlı)
    Füzyon ve zamansal Transformer V1 ile aynı; "feat" çıktısı meta veri adversaryal başlığı için verilir.
    """

    def __init__(self, d=256, streams=("sem", "ftmp"), stream_drop=0.2,
                 sem_backbone="vit_base_patch14_dinov2.lvd142m", **_):
        super().__init__()
        self.streams = streams
        self.stream_drop = stream_drop
        self.sem = timm.create_model(sem_backbone, pretrained=True, num_classes=0, img_size=224)
        for p in self.sem.parameters():
            p.requires_grad = False
        cfg = self.sem.pretrained_cfg
        self.register_buffer("sem_mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("sem_std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        depth = len(self.sem.blocks)
        self.sem_n = [depth // 3 - 1, 2 * depth // 3 - 1, depth - 1]     # ViT-B: 3,7,11
        self.tmp_n = [2 * depth // 3 - 1, depth - 1]                     # ViT-B: 7,11
        ed = self.sem.embed_dim
        if "sem" in streams:
            self.sem_proj = nn.Sequential(nn.LayerNorm(ed * (len(self.sem_n) + 1)),
                                          nn.Linear(ed * (len(self.sem_n) + 1), d), nn.GELU())
        if "ftmp" in streams:
            self.diff_proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(ed), nn.Linear(ed, d), nn.GELU()) for _ in self.tmp_n])
            n_stat = 8
            fin = len(self.tmp_n) * (2 * d + n_stat)
            self.ftmp_proj = nn.Sequential(nn.LayerNorm(fin), nn.Linear(fin, d), nn.GELU())
        if "for" in streams:
            self.srm = SRMResidual()
            self.for_cnn = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
            self.for_proj = nn.Sequential(nn.LayerNorm(1280), nn.Linear(1280, d), nn.GELU())
        k = len(streams)
        self.stream_emb = nn.Parameter(torch.zeros(k, d))
        self.gate = nn.Sequential(nn.Linear(k * d, d), nn.GELU(), nn.Linear(d, k))
        self.pos = nn.Parameter(torch.zeros(1, 17, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, 1))
        self.aux = nn.ModuleDict({s: nn.Linear(d, 1) for s in streams})

    def backbone_tokens(self, x, layers):
        x = (x - self.sem_mean) / self.sem_std
        with torch.no_grad():
            inter = self.sem.get_intermediate_layers(x, n=layers, return_prefix_tokens=True, norm=True)
        return {l: (p.float(), c[:, 0].float()) for l, (p, c) in zip(layers, inter)}   # patch (N,P,D), cls (N,D)

    @staticmethod
    def cos_stats(pt, pt1, ct, ct1):
        c = F.cosine_similarity(pt, pt1, dim=-1)                                       # (N,P)
        q = torch.quantile(c, torch.tensor([0.1, 0.5, 0.9], device=c.device), dim=1).T
        cc = F.cosine_similarity(ct, ct1, dim=-1).unsqueeze(1)
        dn = (ct1 - ct).norm(dim=-1, keepdim=True) / (ct.norm(dim=-1, keepdim=True) + 1e-6)
        return torch.cat([c.mean(1, keepdim=True), c.std(1, keepdim=True), c.min(1, keepdim=True)[0], q, cc, dn], 1)

    def forward(self, x):
        # x: (B,32,H,W,3) -> 16 konum x (t, t+1)
        b = x.shape[0]
        x = to_float(x)
        f, g = x[:, 0::2], x[:, 1::2]
        t = f.shape[1]
        ff, gg = f.flatten(0, 1), g.flatten(0, 1)
        toks = {}
        need_t1 = "ftmp" in self.streams
        layers = sorted(set(self.sem_n) | (set(self.tmp_n) if need_t1 else set()))
        if need_t1:
            both = self.backbone_tokens(torch.cat([ff, gg], 0), layers)
            n = ff.shape[0]
            tok_t = {l: (p[:n], c[:n]) for l, (p, c) in both.items()}
            tok_t1 = {l: (p[n:], c[n:]) for l, (p, c) in both.items()}
        else:
            tok_t = self.backbone_tokens(ff, layers)
        if "sem" in self.streams:
            feats = [tok_t[l][1] for l in self.sem_n] + [tok_t[self.sem_n[-1]][0].mean(1)]
            toks["sem"] = self.sem_proj(torch.cat(feats, -1)).view(b, t, -1)
        if need_t1:
            parts = []
            for j, l in enumerate(self.tmp_n):
                (pt, ct), (pt1, ct1) = tok_t[l], tok_t1[l]
                h = self.diff_proj[j](pt1 - pt)                                        # (N,P,d)
                parts += [h.mean(1), h.amax(1), self.cos_stats(pt, pt1, ct, ct1)]
            toks["ftmp"] = self.ftmp_proj(torch.cat(parts, -1)).view(b, t, -1)
        if "for" in self.streams:
            toks["for"] = self.for_proj(self.for_cnn(self.srm(ff))).view(b, t, -1)
        S = torch.stack([toks[s] + self.stream_emb[i] for i, s in enumerate(self.streams)], 2)   # B,T,K,d
        if self.training and self.stream_drop > 0 and len(self.streams) > 1:
            keep = (torch.rand(b, 1, len(self.streams), 1, device=S.device) > self.stream_drop).float()
            keep[keep.sum(2, keepdim=True).expand_as(keep) == 0] = 1.0
            S = S * keep
        w = torch.softmax(self.gate(S.flatten(2)), -1)                                             # B,T,K
        z = (S * w.unsqueeze(-1)).sum(2)
        z = torch.cat([self.cls.expand(b, -1, -1), z], 1) + self.pos[:, : t + 1]
        z = self.norm(self.temporal(z))
        out = {"logit": self.head(z[:, 0]).squeeze(1), "gates": w.detach(), "feat": z[:, 0]}
        out["aux"] = {s: self.aux[s](toks[s].mean(1)).squeeze(1) for s in self.streams}
        return out


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class MetaAdversary(nn.Module):
    """
    Koşullu meta veri adversaryı: video temsili + sınıf etiketi -> en-boy sınıfı ve çözünürlük sınıfı.
    Etiket girdi olarak verildiği için adversary sınıf-içi meta veri bilgisini cezalandırır;
    temsil sınıf bilgisini korurken kısayolu (çözünürlük/en-boy) taşımamaya zorlanır.
    """

    def __init__(self, d=256, n_heads=(3, 4)):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d + 2, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, d), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(d, n) for n in n_heads])

    def forward(self, feat, y, lam):
        h = _GradReverse.apply(feat.float(), lam)
        h = self.body(torch.cat([h, F.one_hot(y.long(), 2).float()], 1))
        return [hd(h) for hd in self.heads]


# =============================================================================
# 3) Karşılaştırma modelleri
# =============================================================================
class R2Plus1D(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.video.r2plus1d_18(weights=torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1)
        m.fc = nn.Linear(512, 1)
        self.m = m
        self.register_buffer("mean", torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1))

    def forward(self, x):
        x = to_float(x)[:, 0::2] if x.shape[1] == 32 else to_float(x)
        x = F.interpolate(x.flatten(0, 1), size=112, mode="bilinear", antialias=True).view(x.shape[0], x.shape[1], 3, 112, 112)
        x = (x.permute(0, 2, 1, 3, 4) - self.mean) / self.std
        return {"logit": self.m(x).squeeze(1)}


class VideoMAE(nn.Module):
    def __init__(self, name="MCG-NJU/videomae-base-finetuned-kinetics"):
        super().__init__()
        from transformers import VideoMAEForVideoClassification
        self.m = VideoMAEForVideoClassification.from_pretrained(name, num_labels=1, ignore_mismatched_sizes=True)

    def forward(self, x):
        x = to_float(x)[:, 0::2] if x.shape[1] == 32 else to_float(x)
        x = imnet_norm(x.flatten(0, 1)).view(x.shape)
        return {"logit": self.m(pixel_values=x).logits.squeeze(1)}


class FrameCNN(nn.Module):
    """Kare tabanlı klasik deepfake baseline'ı (Xception / EfficientNet), kare logit ortalaması."""

    def __init__(self, name="legacy_xception"):
        super().__init__()
        self.m = timm.create_model(name, pretrained=True, num_classes=1)
        cfg = self.m.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))

    def forward(self, x):
        x = to_float(x)[:, 0::2] if x.shape[1] == 32 else to_float(x)
        b, t = x.shape[:2]
        z = (x.flatten(0, 1) - self.mean) / self.std
        return {"logit": self.m(z).view(b, t).mean(1)}


def build_model(name, **kw):
    if name == "orig":
        return OrigNet()
    if name == "tristream":
        return TriStreamNet(**kw)
    if name == "tristream2":
        return TriStreamV2(**kw)
    if name == "r2plus1d":
        return R2Plus1D()
    if name == "videomae":
        return VideoMAE()
    if name.startswith("frame_"):
        return FrameCNN(name[len("frame_"):])
    raise ValueError(name)
