"""Video captioning — generate natural-language descriptions of videos.

Uses a vision encoder to extract frame features and a language decoder
to produce captions autoregressively.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn

from flashvideo.data.frame_sampler import UniformSampler
from flashvideo.data.transforms import VideoTransform
from flashvideo.data.video_reader import VideoReader
from flashvideo.registry import TASKS


class CaptionDecoder(nn.Module):
    """Lightweight causal transformer decoder for caption generation."""

    def __init__(self, vocab_size: int = 30522, embed_dim: int = 768, depth: int = 6, num_heads: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(512, embed_dim)
        layer = nn.TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.embed(token_ids) + self.pos_embed(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=token_ids.device)
        x = self.decoder(x, memory, tgt_mask=causal_mask)
        return self.head(x)


@TASKS.register("video_captioning")
class VideoCaptioner:
    """Generate captions for videos.

    Args:
        vision_encoder: Model returning features from video frames.
        decoder: Caption decoder (or None to use a built-in stub).
        device: Target device.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        decoder: Optional[nn.Module] = None,
        num_frames: int = 16,
        image_size: int = 224,
        max_length: int = 64,
        device: str = "auto",
    ) -> None:
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
        self.encoder = vision_encoder.to(self.device).eval()
        self.decoder = (decoder or CaptionDecoder()).to(self.device).eval()
        self.transform = VideoTransform(size=image_size)
        self.sampler = UniformSampler(num_frames)
        self.max_length = max_length

    def _extract_features(self, source: Union[str, Path, torch.Tensor]) -> torch.Tensor:
        if isinstance(source, torch.Tensor):
            video = source
        else:
            reader = VideoReader(str(source))
            indices = self.sampler.sample(reader.num_frames)
            frames = reader.get_batch(indices)
            video = self.transform(frames)

        video = video.unsqueeze(0).to(self.device)
        output = self.encoder(video)
        if isinstance(output, dict):
            return output.get("features", output.get("logits")).unsqueeze(1)
        return output.unsqueeze(1) if output.ndim == 2 else output

    @torch.no_grad()
    def caption(self, source: Union[str, Path, torch.Tensor]) -> str:
        """Generate a text caption for the given video."""
        memory = self._extract_features(source)
        tokens = torch.zeros(1, 1, dtype=torch.long, device=self.device)

        generated: List[int] = []
        for _ in range(self.max_length):
            logits = self.decoder(tokens, memory)
            next_token = logits[:, -1].argmax(dim=-1)
            generated.append(next_token.item())
            if next_token.item() == 0:
                break
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)

        return " ".join(str(t) for t in generated)
