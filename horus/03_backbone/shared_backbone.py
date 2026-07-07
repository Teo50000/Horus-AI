from __future__ import annotations


import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple, Union


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor


@dataclass
class BackboneConfig:
    pretrained: bool = False
    input_size: int = 384
    fpn_dim: int = 256
    embed_dim: int = 512
    temporal_hidden: int = 256
    temporal_window: int = 8
    freeze_encoder: bool = False

    novelty_threshold: float = 3.0
    uncertainty_threshold: float = 0.45
    gate_warmup: int = 15
    ema_decay: float = 0.95


_RETURN_NODES = {"layer2": "p3", "layer3": "p4", "layer4": "p5"}
_STRIDES = {"p3": 8, "p4": 16, "p5": 32}


@dataclass
class SharedFeatures:
    pyramid: Dict[str, torch.Tensor]
    embedding: torch.Tensor
    temporal: Optional[torch.Tensor]
    novelty: float
    input_size: Tuple[int, int]
    original_size: Tuple[int, int]
    camera_id: str
    frame_idx: int
    timestamp: float
    strides: Dict[str, int] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:

        return int(self.embedding.shape[0])

    def scale_xy(self, level: str) -> float:

        h_in, _ = self.input_size
        h_org, _ = self.original_size
        return (h_org / h_in) * self.strides.get(level, 1)


class FramePreprocessor(nn.Module):


    def __init__(self, size: int = 384) -> None:
        super().__init__()
        self.size = (size, size)


        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, frame: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[int, int]]:

        t = torch.from_numpy(frame) if isinstance(frame, np.ndarray) else frame


        if t.ndim == 3:
            t = t.permute(2, 0, 1).unsqueeze(0)
        t = t.float()
        original_size = (int(t.shape[-2]), int(t.shape[-1]))
        if t.max() > 1.5:
            t = t / 255.0

        t = F.interpolate(t, size=self.size, mode="bilinear", align_corners=False)

        t = (t - self.mean) / self.std
        return t, original_size


class FeaturePyramid(nn.Module):


    def __init__(self, in_channels: Dict[str, int], out_channels: int) -> None:
        super().__init__()

        self.lateral = nn.ModuleDict({k: nn.Conv2d(c, out_channels, 1) for k, c in in_channels.items()})

        self.smooth = nn.ModuleDict({k: nn.Conv2d(out_channels, out_channels, 3, padding=1) for k in in_channels})

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        laterals = {k: self.lateral[k](v) for k, v in feats.items()}

        order = sorted(laterals, key=lambda k: laterals[k].shape[-1])
        out: Dict[str, torch.Tensor] = {}
        prev: Optional[torch.Tensor] = None


        for name in order:
            cur = laterals[name]
            if prev is not None:
                cur = cur + F.interpolate(prev, size=cur.shape[-2:], mode="nearest")
            prev = cur
            out[name] = self.smooth[name](cur)
        return out


class TemporalAggregator(nn.Module):


    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(dim, hidden, batch_first=True)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(seq)
        return out[:, -1]


class SharedBackbone(nn.Module):


    def __init__(self, config: Optional[BackboneConfig] = None) -> None:
        super().__init__()
        self.cfg = config or BackboneConfig()


        weights = "IMAGENET1K_V2" if self.cfg.pretrained else None
        encoder = resnet50(weights=weights)

        self.extractor = create_feature_extractor(encoder, return_nodes=_RETURN_NODES)
        self.strides = dict(_STRIDES)


        in_ch = self._discover_channels()
        self.fpn = FeaturePyramid(in_ch, self.cfg.fpn_dim)


        self.embed_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.cfg.fpn_dim, self.cfg.embed_dim),
        )

        self.temporal = TemporalAggregator(self.cfg.embed_dim, self.cfg.temporal_hidden)
        self.preprocess = FramePreprocessor(self.cfg.input_size)


        if self.cfg.freeze_encoder:
            self.set_encoder_frozen(True)


        self._buffers: Dict[str, Deque[torch.Tensor]] = defaultdict(
            lambda: deque(maxlen=self.cfg.temporal_window)
        )
        self._counters: Dict[str, int] = defaultdict(int)
        self._stats: Dict[str, Dict[str, Any]] = {}


    def _discover_channels(self) -> Dict[str, int]:

        self.extractor.eval()
        with torch.no_grad():
            feats = self.extractor(torch.zeros(1, 3, self.cfg.input_size, self.cfg.input_size))
        return {k: int(v.shape[1]) for k, v in feats.items()}


    @torch.no_grad()
    def encode(
        self, frame: Union[np.ndarray, torch.Tensor], camera_id: str = "cam0", **kw
    ) -> SharedFeatures:

        x, original = self.preprocess(frame)
        x = x.to(next(self.parameters()).device)
        return self.forward(x, camera_id=camera_id, original_size=original, **kw)

    def forward(
        self,
        x: torch.Tensor,
        camera_id: str = "cam0",
        original_size: Optional[Tuple[int, int]] = None,
        timestamp: Optional[float] = None,
        update_temporal: bool = True,
    ) -> SharedFeatures:

        feats = self.extractor(x)
        pyramid = self.fpn(feats)
        embedding = self.embed_head(pyramid["p5"])


        temporal = self._update_temporal(camera_id, embedding) if update_temporal else None

        novelty = self._update_novelty(camera_id, embedding)


        frame_idx = self._counters[camera_id]
        self._counters[camera_id] += 1


        return SharedFeatures(
            pyramid=pyramid,
            embedding=embedding,
            temporal=temporal,
            novelty=novelty,
            input_size=(int(x.shape[-2]), int(x.shape[-1])),
            original_size=original_size or (int(x.shape[-2]), int(x.shape[-1])),
            camera_id=camera_id,
            frame_idx=frame_idx,
            timestamp=timestamp if timestamp is not None else time.time(),
            strides=self.strides,
        )


    def should_run_vlm(self, features: SharedFeatures, head_uncertainty: float = 0.0) -> bool:

        st = self._stats.get(features.camera_id)

        warm = bool(st and st["seen"] >= self.cfg.gate_warmup)
        by_novelty = warm and features.novelty >= self.cfg.novelty_threshold
        by_doubt = head_uncertainty >= self.cfg.uncertainty_threshold
        return bool(by_novelty or by_doubt)


    def set_encoder_frozen(self, frozen: bool = True) -> None:

        for p in self.extractor.parameters():
            p.requires_grad_(not frozen)

    def reset(self, camera_id: Optional[str] = None) -> None:

        for store in (self._buffers, self._counters, self._stats):
            if camera_id is None:
                store.clear()
            else:
                store.pop(camera_id, None)

    def param_report(self) -> Dict[str, int]:

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


    def _update_temporal(self, camera_id: str, embedding: torch.Tensor) -> Optional[torch.Tensor]:

        buf = self._buffers[camera_id]
        buf.append(embedding[-1:].detach())
        if len(buf) < 2:
            return None
        seq = torch.cat(list(buf), dim=0).unsqueeze(0)
        return self.temporal(seq)

    def _update_novelty(self, camera_id: str, embedding: torch.Tensor) -> float:

        cur = embedding[-1:].detach()
        st = self._stats.get(camera_id)
        if st is None:

            self._stats[camera_id] = {"ema": cur.clone(), "m": 0.0, "v": 1e-3, "seen": 1}
            return 0.0


        d = float((cur - st["ema"]).norm() / (st["ema"].norm() + 1e-6))
        warm = st["seen"] >= self.cfg.gate_warmup

        z = (d - st["m"]) / ((st["v"] ** 0.5) + 1e-6) if warm else 0.0


        if (not warm) or (z < self.cfg.novelty_threshold):
            a = 1.0 - self.cfg.ema_decay
            st["ema"] = self.cfg.ema_decay * st["ema"] + a * cur
            diff = d - st["m"]
            st["m"] += a * diff
            st["v"] = self.cfg.ema_decay * (st["v"] + a * diff * diff)
        st["seen"] += 1
        return max(0.0, z)


VideoBackbone = SharedBackbone
