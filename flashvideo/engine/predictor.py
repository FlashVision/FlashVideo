"""Inference engine for FlashVideo models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

try:
    from flashvideo.data.frame_sampler import UniformSampler
    from flashvideo.data.transforms import VideoTransform
    from flashvideo.data.video_reader import VideoReader
except ImportError:
    UniformSampler = None
    VideoTransform = None
    VideoReader = None


class Predictor:
    """Run inference with a trained FlashVideo model.

    Supports classification, captioning, and embedding extraction.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        num_frames: int = 16,
        image_size: int = 256,
    ) -> None:
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu"
        )
        self.model = model.to(self.device).eval()
        self.transform = VideoTransform(size=image_size)
        self.sampler = UniformSampler(num_frames)

    def _load_video(self, source: Union[str, Path, torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(source, torch.Tensor):
            video = source
        elif isinstance(source, np.ndarray):
            video = torch.from_numpy(source)
        else:
            reader = VideoReader(str(source))
            indices = self.sampler.sample(reader.num_frames)
            video = reader.get_batch(indices)

        if video.ndim == 4:
            video = self.transform(video)
        return video.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, source: Union[str, Path, torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """Run the model on a single video and return raw outputs."""
        video = self._load_video(source)
        output = self.model(video)
        if isinstance(output, dict):
            return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
        return {"output": output.cpu() if isinstance(output, torch.Tensor) else output}

    @torch.no_grad()
    def classify(
        self,
        source: Union[str, Path, torch.Tensor],
        class_names: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[tuple]:
        """Return top-K ``(label, score)`` predictions."""
        video = self._load_video(source)
        output = self.model(video)
        logits = output["logits"] if isinstance(output, dict) else output
        probs = torch.softmax(logits.squeeze(0), dim=-1)
        values, indices = probs.topk(min(top_k, probs.shape[-1]))

        results = []
        for v, i in zip(values.tolist(), indices.tolist()):
            label = class_names[i] if class_names and i < len(class_names) else str(i)
            results.append((label, v))
        return results

    @torch.no_grad()
    def embed(self, source: Union[str, Path, torch.Tensor]) -> torch.Tensor:
        """Extract a feature embedding from the model."""
        video = self._load_video(source)
        output = self.model(video)
        if isinstance(output, dict) and "features" in output:
            return output["features"].cpu()
        return output.cpu() if isinstance(output, torch.Tensor) else torch.tensor(output)
