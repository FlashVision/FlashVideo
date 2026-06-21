"""FlashVideoModel — Unified model wrapper.

Provides a single entry point to build any FlashVideo architecture
(VideoDiT, VideoViT, TimeSformer, WorldModel) from a configuration
dictionary or dataclass.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from flashvideo.registry import MODELS


class FlashVideoModel(nn.Module):
    """Wrapper that constructs a registered model by name and delegates forward calls.

    Usage::

        model = FlashVideoModel(arch="VideoViT", num_classes=400)
        out = model(video_tensor)
    """

    def __init__(self, arch: str = "VideoViT", **kwargs: Any) -> None:
        super().__init__()
        self.arch_name = arch
        self.model = MODELS.build(arch, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def load_pretrained(self, path: str, strict: bool = True) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=strict)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
