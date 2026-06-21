"""Video Diffusion Transformer (Video DiT).

A 3D diffusion transformer for video generation inspired by DiT and
Sora-style architectures.  Operates on patchified latent video tokens with
joint spatial-temporal attention and adaptive layer-norm conditioning on
diffusion timestep and optional text embeddings.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashvideo.registry import MODELS


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding projected through an MLP."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class Attention3D(nn.Module):
    """Multi-head self-attention with optional cross-attention for text conditioning."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.cross_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.cross_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.cross_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, c = x.shape

        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v)
        x = attn.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)

        if context is not None:
            q_c = self.cross_q(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            kv_c = self.cross_kv(context).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k_c, v_c = kv_c.unbind(0)
            cross_attn = F.scaled_dot_product_attention(q_c, k_c, v_c)
            x = x + self.cross_proj(cross_attn.transpose(1, 2).reshape(b, n, c))

        return x


class DiTBlock(nn.Module):
    """A single DiT block with adaptive layer norm, self-attention,
    optional cross-attention, and feed-forward network."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention3D(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        ada = self.adaLN(c).chunk(6, dim=-1)
        shift_attn, scale_attn, gate_attn = ada[0], ada[1], ada[2]
        shift_mlp, scale_mlp, gate_mlp = ada[3], ada[4], ada[5]

        x = x + gate_attn.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_attn, scale_attn), context)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PatchEmbed3D(nn.Module):
    """Convert a video ``(B, C, T, H, W)`` into patch tokens."""

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 768,
        patch_size: tuple = (1, 2, 2),
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T, H, W) -> (B, D, Tp, Hp, Wp) -> (B, N, D)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


@MODELS.register("VideoDiT")
class VideoDiT(nn.Module):
    """Video Diffusion Transformer for Sora/Veo-style video generation.

    Args:
        in_channels: Latent channels (e.g. 4 for VAE latents).
        hidden_size: Transformer hidden dimension.
        depth: Number of DiT blocks.
        num_heads: Attention heads.
        patch_size: 3D patch size ``(t, h, w)``.
        num_frames: Default temporal length for positional embeddings.
        image_size: Spatial resolution.
        context_dim: Text conditioning dimension (None = unconditional).
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: tuple = (1, 2, 2),
        num_frames: int = 16,
        image_size: int = 256,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.out_channels = in_channels

        self.patch_embed = PatchEmbed3D(in_channels, hidden_size, patch_size)

        num_patches_t = num_frames // patch_size[0]
        num_patches_h = image_size // patch_size[1]
        num_patches_w = image_size // patch_size[2]
        self.num_patches = num_patches_t * num_patches_h * num_patches_w
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        self.t_embed = TimestepEmbedding(hidden_size)

        self.context_proj = nn.Linear(context_dim, hidden_size) if context_dim else None

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])

        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.final_proj = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * in_channels)

        self.patch_size = patch_size
        self.num_frames = num_frames
        self.image_size = image_size
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def unpatchify(self, x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        """Convert patch tokens back to video tensor."""
        pt, ph, pw = self.patch_size
        nt, nh, nw = t // pt, h // ph, w // pw
        x = x.reshape(-1, nt, nh, nw, pt, ph, pw, self.in_channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return x.reshape(-1, self.in_channels, t, h, w)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy latent video ``(B, C, T, H, W)``.
            timestep: Diffusion timestep ``(B,)``.
            context: Optional text embeddings ``(B, L, D)``.

        Returns:
            Predicted noise ``(B, C, T, H, W)``.
        """
        _, _, t_len, h, w = x.shape

        x = self.patch_embed(x)
        if x.shape[1] <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, : x.shape[1]]
        else:
            x = x + F.interpolate(
                self.pos_embed.transpose(1, 2), size=x.shape[1], mode="linear", align_corners=False
            ).transpose(1, 2)

        c = self.t_embed(timestep)

        ctx = None
        if context is not None and self.context_proj is not None:
            ctx = self.context_proj(context)

        for block in self.blocks:
            x = block(x, c, ctx)

        ada = self.final_adaLN(c).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), ada[0], ada[1])
        x = self.final_proj(x)

        return self.unpatchify(x, t_len, h, w)
