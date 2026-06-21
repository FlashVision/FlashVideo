"""TimeSformer — Divided Space-Time Attention for Video Understanding.

Implements the *Divided Space-Time* attention pattern where each transformer
block performs temporal attention over all frames at the same spatial position,
followed by spatial attention over all patches within each frame.
"""

from __future__ import annotations


import torch
import torch.nn as nn

from flashvideo.registry import MODELS


class DividedSpaceTimeBlock(nn.Module):
    """Transformer block with factorized temporal then spatial attention."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)

        self.norm_s = nn.LayerNorm(dim)
        self.attn_s = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)

        self.norm_ff = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, nt: int, ns: int) -> torch.Tensor:
        """
        Args:
            x: ``(B, 1 + nt*ns, D)`` — CLS token + space-time patches.
            nt: Number of temporal positions.
            ns: Number of spatial patches per frame.
        """
        b, n, d = x.shape
        cls = x[:, :1]
        tokens = x[:, 1:]  # (B, nt*ns, D)

        # --- Temporal attention ---
        tokens_t = tokens.reshape(b, nt, ns, d)
        tokens_t = tokens_t.permute(0, 2, 1, 3).reshape(b * ns, nt, d)
        h = self.norm_t(tokens_t)
        tokens_t = tokens_t + self.attn_t(h, h, h, need_weights=False)[0]
        tokens = tokens_t.reshape(b, ns, nt, d).permute(0, 2, 1, 3).reshape(b, nt * ns, d)

        # --- Spatial attention ---
        tokens_s = tokens.reshape(b, nt, ns, d).reshape(b * nt, ns, d)
        h = self.norm_s(tokens_s)
        tokens_s = tokens_s + self.attn_s(h, h, h, need_weights=False)[0]
        tokens = tokens_s.reshape(b, nt * ns, d)

        x = torch.cat([cls, tokens], dim=1)

        # --- Feed-forward ---
        x = x + self.mlp(self.norm_ff(x))
        return x


@MODELS.register("TimeSformer")
class TimeSformer(nn.Module):
    """TimeSformer with Divided Space-Time attention.

    Args:
        in_channels: Input channels (3 for RGB).
        num_classes: Number of action classes.
        embed_dim: Transformer hidden dimension.
        depth: Number of transformer blocks.
        num_heads: Attention heads.
        patch_size: Spatial patch size.
        num_frames: Number of input frames.
        image_size: Spatial resolution.
        drop_rate: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 400,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: int = 16,
        num_frames: int = 8,
        image_size: int = 224,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.patch_size = patch_size

        num_spatial = (image_size // patch_size) ** 2
        self.num_spatial = num_spatial
        num_tokens = num_frames * num_spatial

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList(
            [DividedSpaceTimeBlock(embed_dim, num_heads, drop=drop_rate) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video ``(B, C, T, H, W)``.
        """
        b, c, t, h, w = x.shape

        # Per-frame patch embedding
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.patch_embed(x)  # (B*T, D, Hp, Wp)
        ns = x.shape[2] * x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B*T, ns, D)
        x = x.reshape(b, t, ns, -1)

        # Add temporal embedding
        x = x + self.temporal_embed[:, :t].unsqueeze(2)
        x = x.reshape(b, t * ns, -1)

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if x.shape[1] <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, : x.shape[1]]
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x, nt=t, ns=ns)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> dict:
        features = self.forward_features(x)
        logits = self.head(features)
        return {"logits": logits, "features": features}
