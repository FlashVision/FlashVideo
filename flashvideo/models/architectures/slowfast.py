"""SlowFast Networks for Video Recognition.

Implements the dual-pathway architecture where a Slow pathway
operates at low frame rate for spatial semantics, and a Fast
pathway operates at high frame rate for motion dynamics.

Reference: "SlowFast Networks for Video Recognition"
           (Feichtenhofer et al., ICCV 2019)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashvideo.registry import MODELS


class Bottleneck3D(nn.Module):
    """3D ResNet bottleneck block."""

    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        temporal_kernel: int = 1,
        temporal_stride: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(temporal_kernel, 1, 1),
                               stride=(temporal_stride, 1, 1),
                               padding=(temporal_kernel // 2, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3),
                               stride=(1, stride, stride),
                               padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion or temporal_stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=(temporal_stride, stride, stride), bias=False),
                nn.BatchNorm3d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class SlowPathway(nn.Module):
    """Slow pathway — operates at low temporal resolution for spatial semantics."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, layers: Tuple[int, ...] = (3, 4, 6, 3)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=(1, 7, 7),
                               stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(base_channels, base_channels, layers[0])
        self.layer2 = self._make_layer(base_channels * 4, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 8, base_channels * 4, layers[2], stride=2, temporal_kernel=3)
        self.layer4 = self._make_layer(base_channels * 16, base_channels * 8, layers[3], stride=2, temporal_kernel=3)

        self.out_channels = base_channels * 8 * Bottleneck3D.expansion

    def _make_layer(self, in_planes: int, planes: int, blocks: int, stride: int = 1, temporal_kernel: int = 1) -> nn.Sequential:
        layers = [Bottleneck3D(in_planes, planes, stride, temporal_kernel)]
        for _ in range(1, blocks):
            layers.append(Bottleneck3D(planes * Bottleneck3D.expansion, planes, temporal_kernel=temporal_kernel))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class FastPathway(nn.Module):
    """Fast pathway — operates at high temporal resolution for motion."""

    def __init__(self, in_channels: int = 3, base_channels: int = 8, layers: Tuple[int, ...] = (3, 4, 6, 3)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=(5, 7, 7),
                               stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(base_channels, base_channels, layers[0], temporal_kernel=3)
        self.layer2 = self._make_layer(base_channels * 4, base_channels * 2, layers[1], stride=2, temporal_kernel=3)
        self.layer3 = self._make_layer(base_channels * 8, base_channels * 4, layers[2], stride=2, temporal_kernel=3)
        self.layer4 = self._make_layer(base_channels * 16, base_channels * 8, layers[3], stride=2, temporal_kernel=3)

        self.out_channels = base_channels * 8 * Bottleneck3D.expansion

    def _make_layer(self, in_planes: int, planes: int, blocks: int, stride: int = 1, temporal_kernel: int = 1) -> nn.Sequential:
        layers = [Bottleneck3D(in_planes, planes, stride, temporal_kernel)]
        for _ in range(1, blocks):
            layers.append(Bottleneck3D(planes * Bottleneck3D.expansion, planes, temporal_kernel=temporal_kernel))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class LateralConnection(nn.Module):
    """Lateral connection that fuses Fast pathway into Slow pathway via temporal strided conv."""

    def __init__(self, fast_channels: int, slow_channels: int, alpha: int = 8, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Conv3d(
            fast_channels, slow_channels,
            kernel_size=(kernel_size, 1, 1),
            stride=(alpha, 1, 1),
            padding=(kernel_size // 2, 0, 0),
            bias=False,
        )
        self.bn = nn.BatchNorm3d(slow_channels)

    def forward(self, fast_feat: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(fast_feat)), inplace=True)


@MODELS.register("SlowFast")
class SlowFast(nn.Module):
    """SlowFast dual-pathway network for video recognition.

    The Slow pathway captures spatial semantics at low frame rate,
    while the Fast pathway captures temporal dynamics at high frame rate.
    Lateral connections fuse information from Fast to Slow.

    Args:
        in_channels: Input channels (3 for RGB).
        num_classes: Number of action classes.
        slow_channels: Base channels for Slow pathway.
        fast_channels: Base channels for Fast pathway.
        alpha: Temporal stride ratio (Slow samples every alpha-th frame).
        layers: Number of bottleneck blocks per stage.
        drop_rate: Dropout before classification.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 400,
        slow_channels: int = 64,
        fast_channels: int = 8,
        alpha: int = 8,
        layers: Tuple[int, ...] = (3, 4, 6, 3),
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.alpha = alpha

        self.slow = SlowPathway(in_channels, slow_channels, layers)
        self.fast = FastPathway(in_channels, fast_channels, layers)

        self.lateral = LateralConnection(
            self.fast.out_channels, slow_channels * 2, alpha=alpha,
        )

        total_features = self.slow.out_channels + slow_channels * 2
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(total_features, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            x: Video tensor (B, C, T, H, W) — T should be divisible by alpha.

        Returns:
            Dict with 'logits' and 'features'.
        """
        B, C, T, H, W = x.shape

        x_slow = x[:, :, ::self.alpha]
        x_fast = x

        slow_feat = self.slow(x_slow)
        fast_feat = self.fast(x_fast)

        lateral_feat = self.lateral(fast_feat)
        t_slow = slow_feat.shape[2]
        t_lat = lateral_feat.shape[2]
        if t_lat != t_slow:
            lateral_feat = F.interpolate(lateral_feat, size=(t_slow, lateral_feat.shape[3], lateral_feat.shape[4]), mode="trilinear", align_corners=False)

        fused = torch.cat([slow_feat, lateral_feat], dim=1)

        pooled = self.avgpool(fused).flatten(1)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)

        return {"logits": logits, "features": pooled}
