from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import roi_align

@dataclass
class ReIDHeadConfig:
    in_channels: int = 256
    roi_level: str = "p4"
    roi_stride: int = 16
    roi_out: Tuple[int, int] = (16, 8)
    roi_sampling: int = 2

    num_convs: int = 2
    embed_dim: int = 512
    gem_p: float = 3.0

    num_ids: int = 1000
    label_smoothing: float = 0.1

    triplet_margin: float = 0.3
    triplet_weight: float = 1.0
    id_weight: float = 1.0

    @property
    def spatial_scale(self) -> float:
        return 1.0 / float(self.roi_stride)

@dataclass
class BodyEmbedding:
    vector: Tuple[float, ...]
    bbox_xyxy: Tuple[float, float, float, float]
    camera_id: str = "unknown"
    frame_idx: int = -1
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    def as_tensor(self) -> Tensor:
        return torch.tensor(self.vector, dtype=torch.float32)

class GeM(nn.Module):

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / self.p).flatten(1)

class _Tower(nn.Module):
    def __init__(self, channels: int, num_convs: int):
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(num_convs):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ]
        self.tower = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        return self.tower(x)

def _weights_init_classifier(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ReIDBodyHead(nn.Module):
    def __init__(self, cfg: Optional[ReIDHeadConfig] = None):
        super().__init__()
        self.cfg = cfg or ReIDHeadConfig()
        c = self.cfg

        self.tower = _Tower(c.in_channels, c.num_convs)
        self.gem = GeM(c.gem_p)
        self.reduction = nn.Sequential(
            nn.Linear(c.in_channels, c.embed_dim, bias=False),
            nn.BatchNorm1d(c.embed_dim),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.BatchNorm1d(c.embed_dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)

        self.classifier = nn.Linear(c.embed_dim, c.num_ids, bias=False)
        self.classifier.apply(_weights_init_classifier)

    def _roi_pool(
        self,
        fpn_feats: Dict[str, Tensor],
        boxes: Optional[Sequence[Tensor]],
    ) -> Tensor:
        c = self.cfg
        feat = fpn_feats[c.roi_level]

        if boxes is None:
            rois = feat
        else:
            rois = roi_align(
                feat,
                list(boxes),
                output_size=c.roi_out,
                spatial_scale=c.spatial_scale,
                sampling_ratio=c.roi_sampling,
                aligned=True,
            )
        rois = self.tower(rois)
        return self.gem(rois)

    def forward(
        self,
        fpn_feats: Dict[str, Tensor],
        boxes: Optional[Sequence[Tensor]] = None,
    ) -> Dict[str, Tensor]:
        pooled = self._roi_pool(fpn_feats, boxes)
        feat = self.reduction(pooled)
        feat_bn = self.bottleneck(feat)

        out = {"feat": feat, "feat_bn": feat_bn}
        if self.training:
            out["logits"] = self.classifier(feat_bn)
        return out

    @torch.no_grad()
    def embed(
        self,
        fpn_feats: Dict[str, Tensor],
        boxes: Optional[Sequence[Tensor]] = None,
    ) -> Tensor:
        was_training = self.training
        self.eval()
        feat_bn = self.forward(fpn_feats, boxes)["feat_bn"]
        emb = F.normalize(feat_bn, p=2, dim=1)
        if was_training:
            self.train()
        return emb

    @torch.no_grad()
    def embed_people(
        self,
        fpn_feats: Dict[str, Tensor],
        boxes_per_image: Sequence[Tensor],
        camera_ids: Optional[Sequence[str]] = None,
        frame_idxs: Optional[Sequence[int]] = None,
    ) -> List[List[BodyEmbedding]]:
        emb = self.embed(fpn_feats, boxes_per_image).cpu()

        out: List[List[BodyEmbedding]] = []
        offset = 0
        for i, bx in enumerate(boxes_per_image):
            k = int(bx.shape[0])
            cam = camera_ids[i] if camera_ids else "unknown"
            fidx = frame_idxs[i] if frame_idxs else -1
            personas: List[BodyEmbedding] = []
            for j in range(k):
                personas.append(BodyEmbedding(
                    vector=tuple(float(v) for v in emb[offset + j].tolist()),
                    bbox_xyxy=tuple(float(v) for v in bx[j].tolist()),
                    camera_id=cam,
                    frame_idx=fidx,
                ))
            out.append(personas)
            offset += k
        return out

    def compute_loss(
        self,
        fpn_feats: Dict[str, Tensor],
        labels: Tensor,
        boxes: Optional[Sequence[Tensor]] = None,
    ) -> Dict[str, Tensor]:
        c = self.cfg
        out = self.forward(fpn_feats, boxes)

        loss_id = F.cross_entropy(
            out["logits"], labels, label_smoothing=c.label_smoothing
        )
        loss_tri = batch_hard_triplet(out["feat"], labels, c.triplet_margin)

        total = c.id_weight * loss_id + c.triplet_weight * loss_tri
        with torch.no_grad():
            acc = (out["logits"].argmax(1) == labels).float().mean()
        return {"loss_id": loss_id, "loss_triplet": loss_tri,
                "loss": total, "acc": acc}

def _euclidean_dist(x: Tensor, y: Tensor) -> Tensor:
    m, n = x.size(0), y.size(0)
    xx = x.pow(2).sum(1, keepdim=True).expand(m, n)
    yy = y.pow(2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist.addmm(x, y.t(), beta=1, alpha=-2)
    return dist.clamp(min=1e-12).sqrt()

def batch_hard_triplet(feat: Tensor, labels: Tensor, margin: float) -> Tensor:
    dist = _euclidean_dist(feat, feat)
    n = dist.size(0)

    same = labels.expand(n, n).eq(labels.expand(n, n).t())
    diff = ~same

    dist_ap = (dist * same).max(dim=1)[0]
    dist_an = (dist + same.float() * 1e12).min(dim=1)[0]

    if margin > 0:
        return F.relu(dist_ap - dist_an + margin).mean()
    return F.softplus(dist_ap - dist_an).mean()

if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = ReIDHeadConfig(num_ids=50)
    head = ReIDBodyHead(cfg)

    B = 16
    feats = {"p4": torch.randn(B, cfg.in_channels, *cfg.roi_out)}
    labels = torch.randint(0, cfg.num_ids, (B,))
    head.train()
    losses = head.compute_loss(feats, labels)
    print("train:", {k: round(float(v), 4) for k, v in losses.items()})

    head.eval()
    frame_feats = {"p4": torch.randn(1, cfg.in_channels, 24, 24)}
    personas = [torch.tensor([[30., 40., 120., 300.],
                              [150., 20., 240., 320.]])]
    embs = head.embed_people(frame_feats, personas,
                             camera_ids=["cam-7"], frame_idxs=[512])
    print(f"personas en el frame: {len(embs[0])}")
    v = embs[0][0].as_tensor()
    print(f"dim embedding: {v.numel()}  |  norma L2: {v.norm():.4f}")
