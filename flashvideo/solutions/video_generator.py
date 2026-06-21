"""VideoGenerator — High-level API for text-to-video generation.

Abstracts model loading, scheduling, and output saving behind a
single ``generate()`` call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from flashvideo.generation.schedulers import DDIMScheduler
from flashvideo.generation.text_to_video import TextToVideoPipeline
from flashvideo.models.architectures.video_dit import VideoDiT


class VideoGenerator:
    """One-liner video generation.

    Usage::

        gen = VideoGenerator()
        gen.generate("a cat playing in a garden", output="cat.mp4")
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        checkpoint: Optional[str] = None,
        device: str = "auto",
        hidden_size: int = 384,
        depth: int = 6,
        num_heads: int = 6,
    ) -> None:
        self.device = device
        if model is not None:
            self._model = model
        elif checkpoint is not None:
            self._model = self._load(checkpoint, hidden_size, depth, num_heads)
        else:
            self._model = VideoDiT(
                in_channels=4,
                hidden_size=hidden_size,
                depth=depth,
                num_heads=num_heads,
                patch_size=(1, 2, 2),
                num_frames=16,
                image_size=256,
            )

        self.pipeline = TextToVideoPipeline(
            model=self._model,
            scheduler=DDIMScheduler(),
            device=device,
        )

    @staticmethod
    def _load(path: str, hidden_size: int, depth: int, num_heads: int) -> VideoDiT:
        model = VideoDiT(hidden_size=hidden_size, depth=depth, num_heads=num_heads)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        return model

    def generate(
        self,
        prompt: str,
        output: str = "output.mp4",
        num_frames: int = 16,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 256,
        width: int = 256,
        fps: int = 8,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a video and save it.

        Returns:
            Path to the saved video file.
        """
        frames = self.pipeline(
            prompt=prompt,
            num_frames=num_frames,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            seed=seed,
        )
        return self.pipeline.save_video(frames, output, fps=fps)
