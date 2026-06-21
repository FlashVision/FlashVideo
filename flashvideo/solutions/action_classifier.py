"""ActionClassifier — High-level API for video action recognition.

Abstracts backbone loading and classification behind a single call.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch

from flashvideo.models.architectures.timesformer import TimeSformer
from flashvideo.understanding.classification import VideoClassifier


class ActionClassifier:
    """One-liner action classification.

    Usage::

        clf = ActionClassifier()
        results = clf.classify("video.mp4")
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        checkpoint: Optional[str] = None,
        num_classes: int = 400,
        class_names: Optional[List[str]] = None,
        device: str = "auto",
    ) -> None:
        if model is not None:
            backbone = model
        elif checkpoint is not None:
            backbone = TimeSformer(
                num_classes=num_classes, embed_dim=384, depth=6, num_heads=6, num_frames=8, image_size=224
            )
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            backbone.load_state_dict(state, strict=False)
        else:
            backbone = TimeSformer(
                num_classes=num_classes,
                embed_dim=384,
                depth=6,
                num_heads=6,
                num_frames=8,
                image_size=224,
            )

        self._classifier = VideoClassifier(
            model=backbone,
            class_names=class_names,
            num_frames=8,
            image_size=224,
            device=device,
        )

    def classify(
        self,
        video: Union[str, Path, torch.Tensor],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Classify actions in a video. Returns ``[(label, score), ...]``."""
        return self._classifier.classify(video, top_k=top_k)
