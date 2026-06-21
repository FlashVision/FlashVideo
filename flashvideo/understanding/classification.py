"""Video classification / action recognition.

Wraps a backbone (VideoViT, TimeSformer, etc.) with standard classification
heads and provides a high-level predict interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashvideo.data.frame_sampler import UniformSampler
from flashvideo.data.transforms import VideoTransform
from flashvideo.data.video_reader import VideoReader
from flashvideo.registry import TASKS


@TASKS.register("video_classification")
class VideoClassifier:
    """High-level video classification interface.

    Loads a backbone, optionally from a checkpoint, and runs inference
    on raw video files or pre-loaded tensors.
    """

    def __init__(
        self,
        model: nn.Module,
        class_names: Optional[List[str]] = None,
        num_frames: int = 16,
        image_size: int = 224,
        device: str = "auto",
    ) -> None:
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
        self.model = model.to(self.device).eval()
        self.class_names = class_names
        self.transform = VideoTransform(size=image_size)
        self.sampler = UniformSampler(num_frames)

    def _preprocess(self, source: Union[str, Path, torch.Tensor]) -> torch.Tensor:
        if isinstance(source, torch.Tensor):
            if source.ndim == 4:
                source = self.transform(source)
            return source.unsqueeze(0).to(self.device)

        reader = VideoReader(str(source))
        indices = self.sampler.sample(reader.num_frames)
        frames = reader.get_batch(indices)
        video = self.transform(frames)
        return video.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def classify(
        self,
        source: Union[str, Path, torch.Tensor],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return top-K ``(label, probability)`` predictions."""
        video = self._preprocess(source)
        output = self.model(video)
        logits = output["logits"] if isinstance(output, dict) else output
        probs = F.softmax(logits.squeeze(0), dim=-1)

        k = min(top_k, probs.shape[-1])
        values, indices = probs.topk(k)
        results = []
        for v, i in zip(values.tolist(), indices.tolist()):
            label = self.class_names[i] if self.class_names and i < len(self.class_names) else f"class_{i}"
            results.append((label, v))
        return results

    @torch.no_grad()
    def predict_batch(self, sources: List[Union[str, torch.Tensor]], top_k: int = 5) -> List[List[Tuple[str, float]]]:
        """Classify a batch of videos."""
        return [self.classify(s, top_k=top_k) for s in sources]
