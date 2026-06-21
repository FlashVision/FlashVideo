"""VideoMAE — Masked Autoencoder for Video with tube masking.

Implements self-supervised video representation learning via masked
autoencoding with spatiotemporal tube masking and asymmetric
encoder-decoder architecture.

Reference: "VideoMAE: Masked Autoencoders are Data-Efficient Learners
for Self-Supervised Video Pre-Training" (Tong et al., NeurIPS 2022)
"""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

from flashvideo.registry import MODELS


class TubeMasking:
    """Tube masking strategy — masks the same spatial location across all frames."""

    def __init__(self, num_patches_spatial: int, num_frames: int, mask_ratio: float = 0.9):
        self.num_patches_spatial = num_patches_spatial
        self.num_frames = num_frames
        self.mask_ratio = mask_ratio

    def __call__(self, batch_size: int, device: torch.device) -> torch.Tensor:
        num_mask = int(self.num_patches_spatial * self.mask_ratio)
        mask = torch.zeros(batch_size, self.num_patches_spatial, device=device)

        for i in range(batch_size):
            mask_idx = torch.randperm(self.num_patches_spatial, device=device)[:num_mask]
            mask[i, mask_idx] = 1.0

        mask = mask.unsqueeze(1).expand(-1, self.num_frames, -1).reshape(batch_size, -1)
        return mask.bool()


class VideoMAEBlock(nn.Module):
    """Transformer block for VideoMAE encoder/decoder."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.0):
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


class VideoMAEEncoder(nn.Module):
    """Encoder that only processes visible (unmasked) tokens."""

    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            VideoMAEBlock(dim, num_heads, mlp_ratio, drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class VideoMAEDecoder(nn.Module):
    """Lightweight decoder that reconstructs masked tokens."""

    def __init__(self, encoder_dim: int, decoder_dim: int, depth: int, num_heads: int, patch_dim: int):
        super().__init__()
        self.embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList([
            VideoMAEBlock(decoder_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_dim)

    def forward(self, visible_tokens: torch.Tensor, mask: torch.Tensor, num_total: int) -> torch.Tensor:
        B = visible_tokens.shape[0]
        vis = self.embed(visible_tokens)

        full_tokens = self.mask_token.expand(B, num_total, -1).clone()
        visible_idx = (~mask).nonzero(as_tuple=False)
        for b in range(B):
            b_visible = visible_idx[visible_idx[:, 0] == b, 1]
            full_tokens[b, b_visible] = vis[b, :b_visible.shape[0]]

        for block in self.blocks:
            full_tokens = block(full_tokens)

        full_tokens = self.norm(full_tokens)
        return self.head(full_tokens)


@MODELS.register("VideoMAE")
class VideoMAE(nn.Module):
    """VideoMAE: Masked Autoencoder for self-supervised video pre-training.

    Uses tube masking (same spatial mask across frames) with a high
    masking ratio (90%) and an asymmetric encoder-decoder.

    Args:
        in_channels: Input channels (3 for RGB).
        num_classes: Number of classes for fine-tuning (0 for pre-training).
        embed_dim: Encoder hidden dimension.
        encoder_depth: Number of encoder transformer blocks.
        decoder_dim: Decoder hidden dimension.
        decoder_depth: Number of decoder transformer blocks.
        num_heads: Attention heads in encoder.
        decoder_heads: Attention heads in decoder.
        patch_size: Spatial patch size.
        tubelet_length: Temporal extent of each tubelet.
        num_frames: Number of input frames.
        image_size: Spatial input resolution.
        mask_ratio: Fraction of tokens to mask during pre-training.
        drop_rate: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 0,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        decoder_dim: int = 384,
        decoder_depth: int = 4,
        num_heads: int = 12,
        decoder_heads: int = 6,
        patch_size: int = 16,
        tubelet_length: int = 2,
        num_frames: int = 16,
        image_size: int = 224,
        mask_ratio: float = 0.9,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_length = tubelet_length
        self.num_frames = num_frames
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim

        num_spatial = (image_size // patch_size) ** 2
        num_temporal = num_frames // tubelet_length
        self.num_patches = num_spatial * num_temporal
        self.num_spatial = num_spatial
        self.num_temporal = num_temporal
        patch_dim = in_channels * tubelet_length * patch_size * patch_size

        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(tubelet_length, patch_size, patch_size),
            stride=(tubelet_length, patch_size, patch_size),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.encoder = VideoMAEEncoder(embed_dim, encoder_depth, num_heads, drop=drop_rate)

        self.masking = TubeMasking(num_spatial, num_temporal, mask_ratio)

        self.decoder = VideoMAEDecoder(embed_dim, decoder_dim, decoder_depth, decoder_heads, patch_dim)

        self.cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else None

    def _patchify_target(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        tl, ps = self.tubelet_length, self.patch_size
        x = x.reshape(B, C, T // tl, tl, H // ps, ps, W // ps, ps)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, self.num_patches, -1)
        return x

    def forward_pretrain(self, x: torch.Tensor) -> dict:
        """Pre-training forward pass with masking.

        Args:
            x: Video (B, C, T, H, W).

        Returns:
            Dict with 'loss', 'pred', 'target', 'mask'.
        """
        B = x.shape[0]
        target = self._patchify_target(x)

        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1]]

        mask = self.masking(B, x.device)

        visible_tokens = []
        for b in range(B):
            vis_idx = (~mask[b]).nonzero(as_tuple=True)[0]
            visible_tokens.append(tokens[b, vis_idx])
        max_vis = max(vt.shape[0] for vt in visible_tokens)
        padded = torch.zeros(B, max_vis, self.embed_dim, device=x.device)
        for b, vt in enumerate(visible_tokens):
            padded[b, :vt.shape[0]] = vt

        encoded = self.encoder(padded)
        pred = self.decoder(encoded, mask, self.num_patches)

        loss = F.mse_loss(pred[mask.unsqueeze(-1).expand_as(pred)], target[mask.unsqueeze(-1).expand_as(target)])

        return {"loss": loss, "pred": pred, "target": target, "mask": mask}

    def forward_finetune(self, x: torch.Tensor) -> dict:
        """Fine-tuning forward pass (no masking).

        Args:
            x: Video (B, C, T, H, W).

        Returns:
            Dict with 'logits' and 'features'.
        """
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1]]
        features = self.encoder(tokens)
        pooled = features.mean(dim=1)

        logits = self.cls_head(pooled) if self.cls_head is not None else pooled
        return {"logits": logits, "features": pooled}

    def forward(self, x: torch.Tensor, pretrain: bool = False) -> dict:
        if pretrain:
            return self.forward_pretrain(x)
        return self.forward_finetune(x)
