"""CogVideoX — 3D Causal VAE + Expert Transformer for video generation.

Implements the CogVideoX architecture featuring:
- 3D Causal VAE with temporal causal convolutions for latent compression
- Expert Transformer blocks with adaptive LayerNorm conditioning
- Text-video joint attention for text-conditioned video generation

Reference: "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer"
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashvideo.registry import MODELS


class CausalConv3d(nn.Module):
    """3D causal convolution that only attends to past temporal frames."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: Tuple[int, int, int] = (3, 3, 3), stride: Tuple[int, int, int] = (1, 1, 1)):
        super().__init__()
        self.temporal_pad = kernel_size[0] - 1
        spatial_pad = (kernel_size[1] // 2, kernel_size[2] // 2)
        self.conv = nn.Conv3d(
            in_ch, out_ch, kernel_size, stride=stride,
            padding=(0, spatial_pad[0], spatial_pad[1]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, self.temporal_pad, 0))
        return self.conv(x)


class CausalResBlock3D(nn.Module):
    """Residual block with 3D causal convolutions."""

    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, channels), channels)
        self.conv1 = CausalConv3d(channels, channels)
        self.norm2 = nn.GroupNorm(min(groups, channels), channels)
        self.conv2 = CausalConv3d(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class CausalDownsample3D(nn.Module):
    """Temporal-spatial downsampling with causal convolution."""

    def __init__(self, in_ch: int, out_ch: int, temporal_stride: int = 2, spatial_stride: int = 2):
        super().__init__()
        self.conv = CausalConv3d(
            in_ch, out_ch,
            kernel_size=(3, 3, 3),
            stride=(temporal_stride, spatial_stride, spatial_stride),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CausalUpsample3D(nn.Module):
    """Temporal-spatial upsampling followed by causal convolution."""

    def __init__(self, in_ch: int, out_ch: int, temporal_scale: int = 2, spatial_scale: int = 2):
        super().__init__()
        self.temporal_scale = temporal_scale
        self.spatial_scale = spatial_scale
        self.conv = CausalConv3d(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, scale_factor=(self.temporal_scale, self.spatial_scale, self.spatial_scale),
            mode="trilinear", align_corners=False,
        )
        return self.conv(x)


class CausalVAEEncoder(nn.Module):
    """3D Causal VAE Encoder that compresses video into a latent space."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 16, base_channels: int = 128):
        super().__init__()
        self.conv_in = CausalConv3d(in_channels, base_channels)

        self.down1 = nn.Sequential(
            CausalResBlock3D(base_channels),
            CausalResBlock3D(base_channels),
            CausalDownsample3D(base_channels, base_channels * 2),
        )
        self.down2 = nn.Sequential(
            CausalResBlock3D(base_channels * 2),
            CausalResBlock3D(base_channels * 2),
            CausalDownsample3D(base_channels * 2, base_channels * 4),
        )
        self.mid = nn.Sequential(
            CausalResBlock3D(base_channels * 4),
            CausalResBlock3D(base_channels * 4),
        )
        self.norm_out = nn.GroupNorm(32, base_channels * 4)
        self.conv_out = CausalConv3d(base_channels * 4, latent_dim * 2, kernel_size=(1, 1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.mid(h)
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        mu, log_var = h.chunk(2, dim=1)
        return mu, log_var


class CausalVAEDecoder(nn.Module):
    """3D Causal VAE Decoder that reconstructs video from latent space."""

    def __init__(self, out_channels: int = 3, latent_dim: int = 16, base_channels: int = 128):
        super().__init__()
        self.conv_in = CausalConv3d(latent_dim, base_channels * 4, kernel_size=(1, 1, 1))
        self.mid = nn.Sequential(
            CausalResBlock3D(base_channels * 4),
            CausalResBlock3D(base_channels * 4),
        )
        self.up1 = nn.Sequential(
            CausalResBlock3D(base_channels * 4),
            CausalUpsample3D(base_channels * 4, base_channels * 2),
        )
        self.up2 = nn.Sequential(
            CausalResBlock3D(base_channels * 2),
            CausalUpsample3D(base_channels * 2, base_channels),
        )
        self.norm_out = nn.GroupNorm(32, base_channels)
        self.conv_out = CausalConv3d(base_channels, out_channels, kernel_size=(1, 1, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid(h)
        h = self.up1(h)
        h = self.up2(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


class AdaptiveLayerNorm(nn.Module):
    """Adaptive LayerNorm conditioned on timestep embedding."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale_shift = self.proj(cond)
        if scale_shift.dim() == 2:
            scale_shift = scale_shift.unsqueeze(1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class TextVideoJointAttention(nn.Module):
    """Joint attention between text tokens and video tokens."""

    def __init__(self, dim: int, num_heads: int = 8, drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, video_tokens: torch.Tensor, text_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = video_tokens.shape

        if text_tokens is not None:
            kv_tokens = torch.cat([text_tokens, video_tokens], dim=1)
        else:
            kv_tokens = video_tokens

        q = self.q_proj(video_tokens).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_tokens).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_tokens).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class ExpertFeedForward(nn.Module):
    """Expert feed-forward network with gating."""

    def __init__(self, dim: int, mult: float = 4.0, num_experts: int = 4, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mult)
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(hidden, dim),
                nn.Dropout(drop),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_scores = self.gate(x).softmax(dim=-1)
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            output = output + gate_scores[..., i:i+1] * expert(x)
        return output


class ExpertTransformerBlock(nn.Module):
    """Expert Transformer block with adaptive LayerNorm and joint attention."""

    def __init__(self, dim: int, cond_dim: int, num_heads: int = 8, num_experts: int = 4, drop: float = 0.0):
        super().__init__()
        self.adanorm1 = AdaptiveLayerNorm(dim, cond_dim)
        self.attn = TextVideoJointAttention(dim, num_heads, drop)
        self.adanorm2 = AdaptiveLayerNorm(dim, cond_dim)
        self.ffn = ExpertFeedForward(dim, num_experts=num_experts, drop=drop)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, text_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.adanorm1(x, cond), text_tokens)
        x = x + self.ffn(self.adanorm2(x, cond))
        return x


@MODELS.register("CogVideoX")
class CogVideoX(nn.Module):
    """CogVideoX: Expert Transformer for text-to-video diffusion.

    Features a 3D causal VAE for temporal compression and an expert
    transformer denoiser with adaptive LayerNorm and text-video
    joint attention.

    Args:
        in_channels: Latent input channels.
        hidden_size: Transformer hidden dimension.
        depth: Number of expert transformer blocks.
        num_heads: Attention heads.
        num_experts: Number of FFN experts per block.
        patch_size: Spatial-temporal patch for tokenization.
        num_frames: Expected number of latent frames.
        image_size: Spatial resolution of latent.
        context_dim: Text embedding dimension (None disables cross-attn).
        drop_rate: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        num_experts: int = 4,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_frames: int = 16,
        image_size: int = 32,
        context_dim: Optional[int] = 768,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_frames = num_frames

        patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.patch_embed = nn.Linear(patch_dim, hidden_size)

        nt = num_frames // patch_size[0]
        nh = image_size // patch_size[1]
        nw = image_size // patch_size[2]
        num_patches = nt * nh * nw
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        cond_dim = hidden_size
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.text_proj = nn.Linear(context_dim, hidden_size) if context_dim else None

        self.blocks = nn.ModuleList([
            ExpertTransformerBlock(hidden_size, cond_dim, num_heads, num_experts, drop_rate)
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)
        self.final_proj = nn.Linear(hidden_size, patch_dim)

    def _timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.hidden_size // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.hidden_size % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        pt, ph, pw = self.patch_size
        x = x.reshape(B, C, T // pt, pt, H // ph, ph, W // pw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, -1, C * pt * ph * pw)
        return x

    def _unpatchify(self, x: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        B, C, T, H, W = shape
        pt, ph, pw = self.patch_size
        nt, nh, nw = T // pt, H // ph, W // pw
        x = x.reshape(B, nt, nh, nw, C, pt, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B, C, T, H, W)
        return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for denoising.

        Args:
            x: Noisy latent (B, C, T, H, W).
            timesteps: Diffusion timesteps (B,).
            context: Text embeddings (B, L, context_dim).

        Returns:
            Predicted noise (B, C, T, H, W).
        """
        orig_shape = x.shape

        t_emb = self._timestep_embedding(timesteps)
        t_emb = self.time_embed(t_emb)

        text_tokens = None
        if context is not None and self.text_proj is not None:
            text_tokens = self.text_proj(context)

        tokens = self._patchify(x)
        tokens = self.patch_embed(tokens)

        if tokens.shape[1] <= self.pos_embed.shape[1]:
            tokens = tokens + self.pos_embed[:, :tokens.shape[1]]

        for block in self.blocks:
            tokens = block(tokens, t_emb, text_tokens)

        tokens = self.final_norm(tokens)
        tokens = self.final_proj(tokens)

        return self._unpatchify(tokens, orig_shape)
