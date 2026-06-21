"""Video Vision Transformer (ViViT) for video understanding.

Implements a tubelet-embedding Video ViT that jointly models spatial and
temporal information for classification, captioning, and feature extraction.
"""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

from flashvideo.registry import MODELS


class TubeletEmbed(nn.Module):
    """Convert video ``(B, C, T, H, W)`` into tubelet patch tokens."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        tubelet_size: tuple = (2, 16, 16),
    ) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=tubelet_size, stride=tubelet_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class ViTBlock(nn.Module):
    """Standard transformer block with pre-norm, self-attention, and MLP."""

    def __init__(self, dim: int, num_heads: int = 12, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


@MODELS.register("VideoViT")
class VideoViT(nn.Module):
    """Video Vision Transformer for understanding tasks.

    Args:
        in_channels: Input video channels.
        num_classes: Classification head size (0 = feature extractor only).
        embed_dim: Transformer width.
        depth: Number of transformer blocks.
        num_heads: Attention heads.
        tubelet_size: 3D patch dimensions ``(t, h, w)``.
        num_frames: Temporal length for positional embedding init.
        image_size: Spatial resolution.
        drop_rate: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 400,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        tubelet_size: tuple = (2, 16, 16),
        num_frames: int = 16,
        image_size: int = 224,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.patch_embed = TubeletEmbed(in_channels, embed_dim, tubelet_size)

        nt = num_frames // tubelet_size[0]
        nh = image_size // tubelet_size[1]
        nw = image_size // tubelet_size[2]
        num_patches = nt * nh * nw

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads, drop=drop_rate) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token features from video."""
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        if x.shape[1] <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, : x.shape[1]]
        else:
            x = x + F.interpolate(
                self.pos_embed.transpose(1, 2), size=x.shape[1], mode="linear", align_corners=False
            ).transpose(1, 2)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> dict:
        features = self.forward_features(x)
        logits = self.head(features)
        return {"logits": logits, "features": features}
