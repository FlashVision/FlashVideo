"""Frame Interpolation — RIFE-style optical flow-based frame synthesis.

Implements Real-Time Intermediate Flow Estimation for video frame
interpolation, supporting arbitrary timestep interpolation.

Reference: "Real-Time Intermediate Flow Estimation for Video Frame
Interpolation" (Huang et al., ECCV 2022)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warp image using optical flow."""
    B, _, H, W = img.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    grid = grid + flow
    grid_norm = torch.zeros_like(grid)
    grid_norm[:, 0] = 2.0 * grid[:, 0] / (W - 1) - 1.0
    grid_norm[:, 1] = 2.0 * grid[:, 1] / (H - 1) - 1.0
    return F.grid_sample(img, grid_norm.permute(0, 2, 3, 1), mode="bilinear", padding_mode="border", align_corners=True)


class ConvBlock(nn.Module):
    """Conv + PReLU block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.act = nn.PReLU(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class IFBlock(nn.Module):
    """Intermediate Flow estimation block at a single scale."""

    def __init__(self, in_ch: int, mid_ch: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_ch, mid_ch, stride=2),
            ConvBlock(mid_ch, mid_ch),
            ConvBlock(mid_ch, mid_ch, stride=2),
            ConvBlock(mid_ch, mid_ch),
            ConvBlock(mid_ch, mid_ch, stride=2),
            ConvBlock(mid_ch, mid_ch),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(mid_ch, mid_ch, 4, 2, 1),
            nn.PReLU(mid_ch),
            nn.ConvTranspose2d(mid_ch, mid_ch, 4, 2, 1),
            nn.PReLU(mid_ch),
            nn.ConvTranspose2d(mid_ch, mid_ch, 4, 2, 1),
            nn.PReLU(mid_ch),
        )
        self.flow_head = nn.Conv2d(mid_ch, 5, 3, 1, 1)

    def forward(self, x: torch.Tensor, flow: Optional[torch.Tensor] = None, scale: float = 1.0) -> tuple:
        if flow is not None:
            x = torch.cat([x, flow], dim=1)

        if scale != 1.0:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)

        feat = self.encoder(x)
        feat = self.decoder(feat)

        if feat.shape[2:] != x.shape[2:]:
            feat = F.interpolate(feat, size=x.shape[2:], mode="bilinear", align_corners=False)

        out = self.flow_head(feat)

        if scale != 1.0:
            out = F.interpolate(out, scale_factor=scale, mode="bilinear", align_corners=False)
            out[:, :4] *= scale

        flow_out = out[:, :4]
        mask = torch.sigmoid(out[:, 4:5])
        return flow_out, mask


class IFNet(nn.Module):
    """Multi-scale Intermediate Flow Network.

    Estimates bidirectional intermediate flow at multiple scales
    for progressive refinement.
    """

    def __init__(self, in_channels: int = 3, num_scales: int = 3, mid_channels: int = 64):
        super().__init__()
        self.num_scales = num_scales
        base_in = in_channels * 2 + 1

        self.blocks = nn.ModuleList()
        for i in range(num_scales):
            extra = 4 + 1 if i > 0 else 0
            self.blocks.append(IFBlock(base_in + extra, mid_channels))

    def forward(self, img0: torch.Tensor, img1: torch.Tensor, timestep: float = 0.5) -> dict:
        B, C, H, W = img0.shape
        t_map = torch.full((B, 1, H, W), timestep, device=img0.device, dtype=img0.dtype)
        base_input = torch.cat([img0, img1, t_map], dim=1)

        flow = None
        mask = None

        for i, block in enumerate(self.blocks):
            scale = 2 ** (self.num_scales - 1 - i)
            if flow is None:
                flow_delta, mask = block(base_input, scale=scale)
            else:
                flow_input = torch.cat([flow, mask], dim=1)
                flow_delta, mask = block(base_input, flow_input, scale=scale)
            flow = flow_delta if flow is None else flow + flow_delta

        flow_0t = flow[:, :2] * timestep
        flow_1t = flow[:, 2:4] * (1 - timestep)

        warped0 = warp(img0, flow_0t)
        warped1 = warp(img1, flow_1t)

        merged = mask * warped0 + (1 - mask) * warped1
        return {"merged": merged, "flow_0t": flow_0t, "flow_1t": flow_1t, "mask": mask}


class RefineNet(nn.Module):
    """Refinement network that produces the final interpolated frame."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 3 + 4, 64, 3, 1, 1),
            nn.PReLU(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(64),
            nn.Conv2d(64, in_channels, 3, 1, 1),
        )

    def forward(
        self, img0: torch.Tensor, img1: torch.Tensor, merged: torch.Tensor, flow_0t: torch.Tensor, flow_1t: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([img0, img1, merged, flow_0t, flow_1t], dim=1)
        residual = self.net(x)
        return merged + residual


class FrameInterpolator(nn.Module):
    """RIFE-style frame interpolation model.

    Supports interpolating frames at arbitrary intermediate timesteps
    between two input frames using learned intermediate flow estimation.

    Args:
        in_channels: Input image channels (3 for RGB).
        num_scales: Number of flow estimation scales.
        mid_channels: Hidden channels in flow blocks.
    """

    def __init__(self, in_channels: int = 3, num_scales: int = 3, mid_channels: int = 64):
        super().__init__()
        self.ifnet = IFNet(in_channels, num_scales, mid_channels)
        self.refine = RefineNet(in_channels)

    def forward(self, img0: torch.Tensor, img1: torch.Tensor, timestep: float = 0.5) -> dict:
        """Interpolate between two frames.

        Args:
            img0: First frame (B, C, H, W), values in [0, 1].
            img1: Second frame (B, C, H, W), values in [0, 1].
            timestep: Target time (0=img0, 1=img1, 0.5=midpoint).

        Returns:
            Dict with 'output' (interpolated frame), 'flow_0t', 'flow_1t', 'mask'.
        """
        result = self.ifnet(img0, img1, timestep)
        output = self.refine(img0, img1, result["merged"], result["flow_0t"], result["flow_1t"])
        return {
            "output": output.clamp(0, 1),
            "flow_0t": result["flow_0t"],
            "flow_1t": result["flow_1t"],
            "mask": result["mask"],
        }

    @torch.no_grad()
    def interpolate_sequence(self, frames: torch.Tensor, multiplier: int = 2) -> torch.Tensor:
        """Interpolate a video sequence to increase frame rate.

        Args:
            frames: (T, C, H, W) or (B, T, C, H, W) input frames.
            multiplier: Frame rate multiplier (2 doubles FPS).

        Returns:
            Interpolated video with (T-1)*multiplier + 1 frames.
        """
        self.eval()
        squeeze = False
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
            squeeze = True

        B, T, C, H, W = frames.shape
        output_frames = [frames[:, 0]]

        for t in range(T - 1):
            f0 = frames[:, t]
            f1 = frames[:, t + 1]
            for s in range(1, multiplier):
                ts = s / multiplier
                interp = self.forward(f0, f1, ts)["output"]
                output_frames.append(interp)
            output_frames.append(f1)

        result = torch.stack(output_frames, dim=1)
        return result.squeeze(0) if squeeze else result
