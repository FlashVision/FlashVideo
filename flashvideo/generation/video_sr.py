"""Video Super-Resolution with temporal consistency (BasicVSR/EDVR style).

Implements a recurrent video super-resolution network using:
- Bidirectional propagation for temporal feature aggregation
- Optical flow-based alignment via deformable convolution surrogates
- Residual blocks with pixel shuffle upsampling

Reference: "BasicVSR: The Search for Essential Components in Video
Super-Resolution and Beyond" (Chan et al., CVPR 2021)
"""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.lrelu(self.conv1(x))
        res = self.conv2(res)
        return x + res


class FlowEstimator(nn.Module):
    """Lightweight optical flow estimator between adjacent frames."""

    def __init__(self, in_channels: int = 6, mid_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 7, 1, 3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 5, 1, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 5, 1, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, 2, 3, 1, 1),
        )

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([frame1, frame2], dim=1))


def flow_warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp an image/feature using optical flow with bilinear interpolation."""
    B, _, H, W = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=x.dtype),
        torch.arange(W, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    grid = grid + flow.permute(0, 2, 3, 1)
    grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class BidirectionalPropagation(nn.Module):
    """Bidirectional feature propagation along the temporal dimension."""

    def __init__(self, channels: int, num_blocks: int = 5):
        super().__init__()
        self.forward_trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.backward_trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.forward_fuse = nn.Conv2d(channels * 2, channels, 1)
        self.backward_fuse = nn.Conv2d(channels * 2, channels, 1)
        self.flow_estimator = FlowEstimator(in_channels=channels * 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, C, H, W) — per-frame feature maps.

        Returns:
            Propagated features (B, T, C, H, W).
        """
        B, T, C, H, W = features.shape

        forward_feats = [None] * T
        forward_feats[0] = features[:, 0]
        for t in range(1, T):
            flow = self.flow_estimator(features[:, t], features[:, t - 1])
            warped = flow_warp(forward_feats[t - 1], flow)
            fused = self.forward_fuse(torch.cat([features[:, t], warped], dim=1))
            forward_feats[t] = self.forward_trunk(fused)

        backward_feats = [None] * T
        backward_feats[T - 1] = features[:, T - 1]
        for t in range(T - 2, -1, -1):
            flow = self.flow_estimator(features[:, t], features[:, t + 1])
            warped = flow_warp(backward_feats[t + 1], flow)
            fused = self.backward_fuse(torch.cat([features[:, t], warped], dim=1))
            backward_feats[t] = self.backward_trunk(fused)

        output = torch.stack([
            forward_feats[t] + backward_feats[t] for t in range(T)
        ], dim=1)
        return output


class PixelShuffleUpsampler(nn.Module):
    """Pixel-shuffle based spatial upsampler."""

    def __init__(self, in_channels: int, out_channels: int = 3, scale: int = 4):
        super().__init__()
        layers = []
        current_ch = in_channels
        remaining = scale
        while remaining > 1:
            factor = min(2, remaining)
            layers.extend([
                nn.Conv2d(current_ch, current_ch * factor * factor, 3, 1, 1),
                nn.PixelShuffle(factor),
                nn.LeakyReLU(0.1, inplace=True),
            ])
            remaining //= factor
        layers.append(nn.Conv2d(current_ch, out_channels, 3, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VideoSuperResolution(nn.Module):
    """Temporal-consistent video super-resolution (BasicVSR-style).

    Uses bidirectional propagation with flow-based alignment
    and pixel-shuffle upsampling.

    Args:
        in_channels: Input channels (3 for RGB).
        mid_channels: Internal feature channels.
        num_blocks: Residual blocks per propagation direction.
        scale_factor: Upscaling factor (2 or 4).
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 64,
        num_blocks: int = 5,
        scale_factor: int = 4,
    ):
        super().__init__()
        self.scale_factor = scale_factor

        self.feat_extract = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(mid_channels),
            ResidualBlock(mid_channels),
        )

        self.propagation = BidirectionalPropagation(mid_channels, num_blocks)

        self.reconstruction = nn.Sequential(
            ResidualBlock(mid_channels),
            ResidualBlock(mid_channels),
        )

        self.upsample = PixelShuffleUpsampler(mid_channels, in_channels, scale_factor)

    def forward(self, lr_video: torch.Tensor) -> torch.Tensor:
        """Super-resolve a low-resolution video.

        Args:
            lr_video: (B, T, C, H, W) or (B, C, T, H, W) low-res video.

        Returns:
            (B, T, C, H*scale, W*scale) super-resolved video.
        """
        if lr_video.dim() == 5 and lr_video.shape[1] == 3:
            lr_video = lr_video.permute(0, 2, 1, 3, 4)
        B, T, C, H, W = lr_video.shape

        features = []
        for t in range(T):
            features.append(self.feat_extract(lr_video[:, t]))
        features = torch.stack(features, dim=1)

        propagated = self.propagation(features)

        sr_frames = []
        for t in range(T):
            recon = self.reconstruction(propagated[:, t])
            sr = self.upsample(recon)
            upsampled_lr = F.interpolate(lr_video[:, t], scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
            sr_frames.append(sr + upsampled_lr)

        return torch.stack(sr_frames, dim=1)

    @torch.no_grad()
    def enhance(self, video: torch.Tensor) -> torch.Tensor:
        """Convenience inference method (no grad, eval mode).

        Args:
            video: (B, T, C, H, W) low-resolution video, values in [0, 1].

        Returns:
            (B, T, C, H*scale, W*scale) enhanced video clamped to [0, 1].
        """
        self.eval()
        return self.forward(video).clamp(0, 1)
