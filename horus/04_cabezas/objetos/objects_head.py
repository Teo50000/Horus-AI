# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.ops import boxes as box_ops
from torchvision.ops import sigmoid_focal_loss
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection._utils import BoxCoder, Matcher


DEFAULT_CLASSES: Tuple[str, ...] = (
    "humo", "llama", "persona", "pistola", "cuchillo", "celular", "paquete"
)

CRITICAL_CLASSES: Tuple[str, ...] = ("humo", "llama", "pistola", "cuchillo")


@dataclass
class ObjectsHeadConfig:
    classes: Tuple[str, ...] = DEFAULT_CLASSES

    in_channels: int = 256
    num_convs: int = 4
    prior_prob: float = 0.01

    fpn_levels: Tuple[str, ...] = ("p3", "p4", "p5", "p6", "p7")

    anchor_sizes: Tuple[Tuple[int, ...], ...] = (
        (32,), (64,), (128,), (256,), (512,)
    )
    aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0)

    score_thresh: float = 0.30
    nms_thresh: float = 0.50
    detections_per_img: int = 100
    topk_candidates: int = 1000

    vlm_gate_band: Tuple[float, float] = (0.30, 0.60)

    fg_iou_thresh: float = 0.5
    bg_iou_thresh: float = 0.4
    box_loss_weight: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def num_anchors(self) -> int:
        return len(self.aspect_ratios) * len(self.anchor_sizes[0])


@dataclass
class Detection:
    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    class_id: int
    label: str
    camera_id: str = "unknown"
    frame_idx: int = -1
    ts: float = field(default_factory=time.time)
    needs_vlm: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class _Tower(nn.Module):
    def __init__(self, channels: int, num_convs: int):
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(num_convs):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(32, channels),
                nn.ReLU(inplace=True),
            ]
        self.tower = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.tower(x)


class ObjectsHead(nn.Module):
    def __init__(self, cfg: Optional[ObjectsHeadConfig] = None):
        super().__init__()
        self.cfg = cfg or ObjectsHeadConfig()
        c = self.cfg

        self.cls_tower = _Tower(c.in_channels, c.num_convs)
        self.box_tower = _Tower(c.in_channels, c.num_convs)

        self.cls_logits = nn.Conv2d(
            c.in_channels, c.num_anchors * c.num_classes, 3, padding=1
        )
        self.bbox_pred = nn.Conv2d(c.in_channels, c.num_anchors * 4, 3, padding=1)

        nn.init.normal_(self.cls_logits.weight, std=0.01)
        nn.init.constant_(
            self.cls_logits.bias,
            -math.log((1 - c.prior_prob) / c.prior_prob),
        )
        nn.init.normal_(self.bbox_pred.weight, std=0.01)
        nn.init.zeros_(self.bbox_pred.bias)

        self.anchor_generator = AnchorGenerator(
            sizes=c.anchor_sizes,
            aspect_ratios=(c.aspect_ratios,) * len(c.anchor_sizes),
        )
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.matcher = Matcher(
            c.fg_iou_thresh, c.bg_iou_thresh, allow_low_quality_matches=True
        )

    def forward(self, fpn_feats: Dict[str, Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        cls_out, box_out = [], []
        for level in self.cfg.fpn_levels:
            f = fpn_feats[level]
            cls_out.append(self.cls_logits(self.cls_tower(f)))
            box_out.append(self.bbox_pred(self.box_tower(f)))
        return cls_out, box_out

    def _anchors(
        self,
        fpn_feats: Dict[str, Tensor],
        image_sizes: Sequence[Tuple[int, int]],
    ) -> List[Tensor]:
        feats = [fpn_feats[l] for l in self.cfg.fpn_levels]
        device = feats[0].device

        class _ImgList:
            def __init__(self, sizes, dev, batch):
                self.image_sizes = list(sizes)
                h = max(s[0] for s in sizes)
                w = max(s[1] for s in sizes)
                self.tensors = torch.zeros((batch, 3, h, w), device=dev)

        il = _ImgList(image_sizes, device, feats[0].shape[0])
        return self.anchor_generator(il, feats)

    @staticmethod
    def _flatten(per_level: List[Tensor], last_dim: int) -> Tensor:
        out = []
        for t in per_level:
            b, _, h, w = t.shape
            t = t.view(b, -1, last_dim, h, w).permute(0, 3, 4, 1, 2)
            out.append(t.reshape(b, -1, last_dim))
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def predict(
        self,
        fpn_feats: Dict[str, Tensor],
        image_sizes: Sequence[Tuple[int, int]],
        camera_ids: Optional[Sequence[str]] = None,
        frame_idxs: Optional[Sequence[int]] = None,
    ) -> List[List[Detection]]:
        c = self.cfg

        cls_out, box_out = self.forward(fpn_feats)
        anchors = self._anchors(fpn_feats, image_sizes)

        logits = self._flatten(cls_out, c.num_classes)
        deltas = self._flatten(box_out, 4)

        scores_all = logits.sigmoid()

        batch: List[List[Detection]] = []
        for i, (h, w) in enumerate(image_sizes):
            scores = scores_all[i]

            boxes = self.box_coder.decode_single(deltas[i], anchors[i])
            boxes = box_ops.clip_boxes_to_image(boxes, (h, w))

            flat = scores.flatten()
            keep = flat > c.score_thresh
            idx = torch.nonzero(keep).squeeze(1)
            if idx.numel() > c.topk_candidates:
                topv, topi = flat[idx].topk(c.topk_candidates)
                idx = idx[topi]
            anchor_idx = idx // c.num_classes
            class_idx = idx % c.num_classes

            b = boxes[anchor_idx]
            s = flat[idx]
            keep_nms = box_ops.batched_nms(b, s, class_idx, c.nms_thresh)
            keep_nms = keep_nms[: c.detections_per_img]

            cam = camera_ids[i] if camera_ids else "unknown"
            fidx = frame_idxs[i] if frame_idxs else -1
            dets: List[Detection] = []
            for j in keep_nms.tolist():
                cid = int(class_idx[j])
                label = c.classes[cid]
                score = float(s[j])
                dets.append(Detection(
                    bbox_xyxy=tuple(float(v) for v in b[j].tolist()),
                    score=score,
                    class_id=cid,
                    label=label,
                    camera_id=cam,
                    frame_idx=fidx,
                    needs_vlm=self._gate_to_vlm(label, score),
                ))
            batch.append(dets)
        return batch

    def _gate_to_vlm(self, label: str, score: float) -> bool:
        lo, hi = self.cfg.vlm_gate_band
        return label in CRITICAL_CLASSES and lo <= score <= hi

    def compute_loss(
        self,
        fpn_feats: Dict[str, Tensor],
        image_sizes: Sequence[Tuple[int, int]],
        targets: Sequence[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        c = self.cfg

        cls_out, box_out = self.forward(fpn_feats)
        anchors = self._anchors(fpn_feats, image_sizes)
        logits = self._flatten(cls_out, c.num_classes)
        deltas = self._flatten(box_out, 4)

        cls_losses, box_losses = [], []
        for i, tgt in enumerate(targets):
            anchors_i, logits_i, deltas_i = anchors[i], logits[i], deltas[i]
            gt_boxes, gt_labels = tgt["boxes"], tgt["labels"]

            if gt_boxes.numel() == 0:
                gt_cls = torch.zeros_like(logits_i)
                cls_losses.append(sigmoid_focal_loss(
                    logits_i, gt_cls, alpha=c.focal_alpha,
                    gamma=c.focal_gamma, reduction="sum",
                ) / max(1, anchors_i.shape[0]))
                box_losses.append(deltas_i.sum() * 0.0)
                continue

            match = self.matcher(box_ops.box_iou(gt_boxes, anchors_i))
            fg = match >= 0
            valid = match != Matcher.BETWEEN_THRESHOLDS

            gt_cls = torch.zeros_like(logits_i)
            gt_cls[fg, gt_labels[match[fg].clamp(min=0)]] = 1.0
            cls_losses.append(sigmoid_focal_loss(
                logits_i[valid], gt_cls[valid], alpha=c.focal_alpha,
                gamma=c.focal_gamma, reduction="sum",
            ) / max(1, int(fg.sum())))

            if fg.any():
                matched = gt_boxes[match[fg]]
                tgt_deltas = self.box_coder.encode_single(matched, anchors_i[fg])
                box_losses.append(nn.functional.smooth_l1_loss(
                    deltas_i[fg], tgt_deltas, beta=0.11, reduction="mean",
                ))
            else:
                box_losses.append(deltas_i.sum() * 0.0)

        return {
            "loss_cls": torch.stack(cls_losses).mean(),
            "loss_box": torch.stack(box_losses).mean() * c.box_loss_weight,
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = ObjectsHeadConfig()
    head = ObjectsHead(cfg).eval()

    sizes = {"p3": 64, "p4": 32, "p5": 16, "p6": 8, "p7": 4}
    feats = {k: torch.randn(1, cfg.in_channels, s, s) for k, s in sizes.items()}

    dets = head.predict(feats, image_sizes=[(512, 512)],
                        camera_ids=["cam-3"], frame_idxs=[1042])
    print(f"detecciones: {len(dets[0])}")
    for d in dets[0][:5]:
        print(d.to_dict())

    targets = [{
        "boxes": torch.tensor([[50.0, 60.0, 200.0, 220.0]]),
        "labels": torch.tensor([0]),
    }]
    head.train()
    losses = head.compute_loss(feats, [(512, 512)], targets)
    print({k: float(v.detach()) for k, v in losses.items()})
