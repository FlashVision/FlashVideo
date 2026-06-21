"""Visualization helpers for video and attention maps."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def visualize_video_grid(
    videos: torch.Tensor,
    nrow: int = 4,
    padding: int = 2,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Create a grid image from a batch of video first-frames.

    Args:
        videos: ``(B, C, T, H, W)`` or ``(B, T, H, W, C)``.
        nrow: Number of videos per row.
        padding: Pixel padding between grid cells.
        output_path: If given, save the grid as PNG.

    Returns:
        Grid image as ``(H, W, 3)`` uint8 numpy array.
    """
    if videos.ndim == 5 and videos.shape[1] in (1, 3):
        frames = videos[:, :, 0].permute(0, 2, 3, 1)
    elif videos.ndim == 5:
        frames = videos[:, 0]
    else:
        frames = videos

    if frames.is_floating_point():
        frames = ((frames - frames.min()) / (frames.max() - frames.min() + 1e-8) * 255).to(torch.uint8)

    frames = frames.cpu().numpy()
    b, h, w, c = frames.shape
    ncol = nrow
    nrows = (b + ncol - 1) // ncol

    grid_h = nrows * h + (nrows + 1) * padding
    grid_w = ncol * w + (ncol + 1) * padding
    grid = np.zeros((grid_h, grid_w, c), dtype=np.uint8)

    for idx in range(b):
        row = idx // ncol
        col = idx % ncol
        y = padding + row * (h + padding)
        x = padding + col * (w + padding)
        grid[y : y + h, x : x + w] = frames[idx]

    if output_path:
        from PIL import Image

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(grid).save(output_path)

    return grid


def visualize_attention(
    attention_map: torch.Tensor,
    output_path: Optional[str] = None,
    cmap: str = "viridis",
) -> np.ndarray:
    """Visualize a 2D attention map as a heatmap.

    Args:
        attention_map: ``(H, W)`` or ``(N, N)`` attention weights.
        output_path: If given, save the heatmap.
        cmap: Matplotlib colormap name.

    Returns:
        Heatmap image as ``(H, W, 3)`` uint8.
    """
    attn = attention_map.detach().cpu().float().numpy()
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    try:
        import matplotlib.pyplot as plt

        cm = plt.get_cmap(cmap)
        heatmap = (cm(attn)[:, :, :3] * 255).astype(np.uint8)
    except ImportError:
        heatmap = (np.stack([attn] * 3, axis=-1) * 255).astype(np.uint8)

    if output_path:
        from PIL import Image

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(heatmap).save(output_path)

    return heatmap
