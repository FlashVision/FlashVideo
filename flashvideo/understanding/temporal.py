"""Temporal modeling and event detection.

Provides temporal segmentation (action boundaries), event detection
(moments of interest), and temporal feature pooling.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalModeling(nn.Module):
    """Temporal feature aggregation with multi-scale 1D convolutions.

    Takes per-frame features ``(B, T, D)`` and produces temporally-aware
    representations with information across multiple timescales.
    """

    def __init__(self, dim: int = 768, scales: Tuple[int, ...] = (3, 5, 7)) -> None:
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(dim, dim, k, padding=k // 2) for k in scales])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim * len(scales), dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Per-frame features ``(B, T, D)``.

        Returns:
            Temporally-enhanced features ``(B, T, D)``.
        """
        x_t = x.transpose(1, 2)
        feats = [conv(x_t).transpose(1, 2) for conv in self.convs]
        combined = torch.cat(feats, dim=-1)
        return self.norm(self.proj(combined))


class EventDetector(nn.Module):
    """Detect temporal events / action boundaries in video features.

    Produces per-frame event scores and boundary predictions.
    """

    def __init__(self, dim: int = 768, num_classes: int = 1) -> None:
        super().__init__()
        self.temporal = TemporalModeling(dim)
        self.event_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_classes),
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 2),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Per-frame features ``(B, T, D)``.

        Returns:
            Dictionary with ``"event_scores"`` ``(B, T, C)`` and
            ``"boundary_scores"`` ``(B, T, 2)`` (start/end logits).
        """
        enhanced = self.temporal(features)
        return {
            "event_scores": self.event_head(enhanced),
            "boundary_scores": self.boundary_head(enhanced),
        }

    def detect(self, features: torch.Tensor, threshold: float = 0.5) -> List[List[Tuple[int, int, float]]]:
        """Return detected event segments as ``(start_frame, end_frame, score)``."""
        out = self.forward(features)
        event_probs = torch.sigmoid(out["event_scores"].squeeze(-1))
        boundary_probs = torch.sigmoid(out["boundary_scores"])

        results = []
        for b in range(features.shape[0]):
            events = []
            active = False
            start = 0
            for t in range(features.shape[1]):
                if event_probs[b, t] > threshold and not active:
                    active = True
                    start = t
                elif event_probs[b, t] <= threshold and active:
                    active = False
                    score = event_probs[b, start:t].mean().item()
                    events.append((start, t, score))
            if active:
                events.append((start, features.shape[1], event_probs[b, start:].mean().item()))
            results.append(events)
        return results
