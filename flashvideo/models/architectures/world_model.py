"""World Model Transformer — Physics-aware video prediction.

A causal transformer that models environment dynamics conditioned on actions,
inspired by NVIDIA Cosmos and GAIA-1.  It autoregressively predicts future
latent frames given past context and (optional) action embeddings.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashvideo.registry import MODELS


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking for autoregressive prediction."""

    def __init__(self, dim: int, num_heads: int = 8, drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = attn.transpose(1, 2).reshape(b, n, c)
        return self.drop(self.proj(x))


class WorldModelBlock(nn.Module):
    """Causal transformer block with optional action-conditioned gating."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )
        self.action_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor, action_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        h = self.mlp(self.norm2(x))
        if action_emb is not None:
            h = h * self.action_gate(action_emb)
        x = x + h
        return x


@MODELS.register("WorldModel")
class WorldModelTransformer(nn.Module):
    """Autoregressive world model for environment simulation.

    Takes a sequence of latent frame tokens (and optional action embeddings)
    and predicts the next frame's latent representation.

    Args:
        frame_dim: Dimension of each latent frame token.
        hidden_size: Transformer hidden dimension.
        depth: Number of causal transformer blocks.
        num_heads: Attention heads.
        action_dim: Action embedding dimension (0 = unconditional).
        max_frames: Maximum sequence length.
        drop_rate: Dropout probability.
    """

    def __init__(
        self,
        frame_dim: int = 512,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        action_dim: int = 0,
        max_frames: int = 64,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.frame_proj = nn.Linear(frame_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_size))
        self.pos_drop = nn.Dropout(drop_rate)

        self.action_proj = nn.Linear(action_dim, hidden_size) if action_dim > 0 else None

        self.blocks = nn.ModuleList([WorldModelBlock(hidden_size, num_heads, drop=drop_rate) for _ in range(depth)])

        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, frame_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        frames: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            frames: Sequence of latent frames ``(B, T, frame_dim)``.
            actions: Optional action embeddings ``(B, T, action_dim)``.

        Returns:
            Dictionary with ``"predicted_frames"`` of shape ``(B, T, frame_dim)``
            and ``"loss"`` when ground-truth next frames are available (shifted).
        """
        b, t, _ = frames.shape
        x = self.frame_proj(frames)
        x = x + self.pos_embed[:, :t]
        x = self.pos_drop(x)

        action_emb = None
        if actions is not None and self.action_proj is not None:
            action_emb = self.action_proj(actions)

        for block in self.blocks:
            x = block(x, action_emb)

        x = self.norm(x)
        predicted = self.head(x)

        result = {"predicted_frames": predicted}

        if t > 1:
            target = frames[:, 1:]
            pred = predicted[:, :-1]
            result["loss"] = F.mse_loss(pred, target)

        return result

    @torch.no_grad()
    def rollout(
        self,
        initial_frames: torch.Tensor,
        num_steps: int,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressively generate *num_steps* future frames.

        Args:
            initial_frames: Context frames ``(B, T0, frame_dim)``.
            num_steps: Number of future frames to predict.
            actions: Optional actions for each future step ``(B, num_steps, action_dim)``.

        Returns:
            All frames (context + generated) ``(B, T0 + num_steps, frame_dim)``.
        """
        frames = initial_frames
        for i in range(num_steps):
            act = None
            if actions is not None:
                act_step = actions[:, i : i + 1]
                act_ctx = torch.zeros(frames.shape[0], frames.shape[1] - 1, actions.shape[-1], device=frames.device)
                act = torch.cat([act_ctx, act_step], dim=1)

            out = self.forward(frames, act)
            next_frame = out["predicted_frames"][:, -1:]
            frames = torch.cat([frames, next_frame], dim=1)
        return frames
