"""Temporal grounding — locate moments in video matching a text query.

Given a video and a natural-language query, predict the start and end
timestamps of the most relevant segment.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGrounding(nn.Module):
    """Predict temporal spans in a video that correspond to a text query.

    Consumes per-frame visual features and a text embedding, then outputs
    start/end probabilities over the temporal dimension.

    Args:
        visual_dim: Dimension of per-frame visual features.
        text_dim: Dimension of query text embedding.
        hidden_dim: Internal projection dimension.
    """

    def __init__(self, visual_dim: int = 768, text_dim: int = 768, hidden_dim: int = 256) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        self.start_head = nn.Linear(hidden_dim, 1)
        self.end_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_features: ``(B, T, visual_dim)`` per-frame features.
            text_features: ``(B, L, text_dim)`` or ``(B, text_dim)`` query embedding.

        Returns:
            ``"start_logits"`` and ``"end_logits"`` each ``(B, T)``.
        """
        v = self.visual_proj(visual_features)

        if text_features.ndim == 2:
            text_features = text_features.unsqueeze(1)
        t = self.text_proj(text_features)

        fused, _ = self.cross_attn(v, t, t)
        fused = self.norm(fused + v)

        start_logits = self.start_head(fused).squeeze(-1)
        end_logits = self.end_head(fused).squeeze(-1)

        return {"start_logits": start_logits, "end_logits": end_logits}

    def ground(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        fps: float = 30.0,
    ) -> List[Tuple[float, float, float]]:
        """Return ``(start_sec, end_sec, confidence)`` for each batch element."""
        out = self.forward(visual_features, text_features)
        start_probs = F.softmax(out["start_logits"], dim=-1)
        end_probs = F.softmax(out["end_logits"], dim=-1)

        results = []
        for b in range(visual_features.shape[0]):
            s = start_probs[b].argmax().item()
            e = end_probs[b].argmax().item()
            if e < s:
                s, e = e, s
            conf = (start_probs[b, s] * end_probs[b, e]).item()
            results.append((s / fps, e / fps, conf))
        return results
