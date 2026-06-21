"""I/O utilities for checkpoints and video frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    path: str = "checkpoint.pth",
    **extra: Any,
) -> str:
    """Save a training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    state.update(extra)
    torch.save(state, path)
    return path


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint and optionally restore model/optimizer state."""
    state = torch.load(path, map_location=device, weights_only=True)
    if model is not None and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state


def save_video_frames(
    frames: torch.Tensor,
    output_dir: str = "frames/",
    prefix: str = "frame",
) -> str:
    """Save video frames as individual images.

    Args:
        frames: ``(T, H, W, 3)`` uint8 tensor.
        output_dir: Directory to save frames.
        prefix: Filename prefix.

    Returns:
        Path to the output directory.
    """
    from PIL import Image

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame.numpy() if isinstance(frame, torch.Tensor) else frame)
        img.save(out / f"{prefix}_{i:04d}.png")
    return str(out)
