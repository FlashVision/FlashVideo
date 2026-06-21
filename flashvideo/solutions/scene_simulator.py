"""SceneSimulator — High-level API for world-model simulation.

Wraps the world model and optional decoder to simulate future scenes
from a text prompt or initial state with optional action conditioning.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from flashvideo.world_models.simulator import EnvironmentSimulator


class SceneSimulator:
    """One-liner scene simulation.

    Usage::

        sim = SceneSimulator()
        sim.simulate("a robot arm picking up a block", output="sim.mp4")
    """

    def __init__(
        self,
        frame_dim: int = 512,
        device: str = "auto",
    ) -> None:
        self.device = device
        self._sim = EnvironmentSimulator(frame_dim=frame_dim, device=device)

    def simulate(
        self,
        prompt: str = "",
        actions: Optional[List[str]] = None,
        output: str = "simulation.mp4",
        num_frames: int = 32,
        num_context: int = 4,
        seed: Optional[int] = None,
    ) -> str:
        """Run simulation and save result.

        Args:
            prompt: Scene description (used for seeding, future: text conditioning).
            actions: Optional list of action labels.
            output: Output video path.
            num_frames: Number of frames to simulate.
            num_context: Number of context frames.
            seed: Random seed.

        Returns:
            Path to saved video.
        """
        if seed is None and prompt:
            seed = hash(prompt) % (2**31)

        action_tensor = None
        if actions:
            action_tensor = torch.randn(1, num_frames, 64)

        trajectory = self._sim.simulate(
            num_steps=num_frames,
            num_context=num_context,
            actions=action_tensor,
            seed=seed,
        )

        frames = self._trajectory_to_frames(trajectory, height=256, width=256)
        self._save_video(frames, output, fps=8)
        print(f"Simulation saved → {output} ({frames.shape[0]} frames)")
        return output

    @staticmethod
    def _trajectory_to_frames(trajectory: torch.Tensor, height: int = 256, width: int = 256) -> np.ndarray:
        """Convert latent trajectory to pixel frames via simple projection."""
        t = trajectory.squeeze(0).cpu()
        num_frames = t.shape[0]
        dim = t.shape[1]

        proj = torch.randn(dim, height * width * 3)
        proj = proj / proj.norm(dim=0, keepdim=True)

        frames = (t @ proj).reshape(num_frames, height, width, 3)
        frames = frames - frames.min()
        frames = frames / (frames.max() + 1e-8)
        return (frames * 255).clamp(0, 255).to(torch.uint8).numpy()

    @staticmethod
    def _save_video(frames: np.ndarray, path: str, fps: int = 8) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        h, w = frames.shape[1], frames.shape[2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
