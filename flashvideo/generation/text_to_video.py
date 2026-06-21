"""Text-to-video generation pipeline.

Generates video frames from a text prompt using a Video DiT denoiser
with iterative denoising through a configurable scheduler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from flashvideo.generation.schedulers import BaseScheduler, DDIMScheduler


class TextToVideoPipeline:
    """End-to-end text-to-video generation.

    Args:
        model: A ``VideoDiT`` (or compatible) denoising model.
        scheduler: Noise scheduler instance.
        text_encoder: Optional text encoder producing ``(B, L, D)`` embeddings.
        vae_decoder: Optional VAE decoder to convert latents to pixels.
        device: Target device.
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler: Optional[BaseScheduler] = None,
        text_encoder: Optional[nn.Module] = None,
        vae_decoder: Optional[nn.Module] = None,
        device: str = "auto",
    ) -> None:
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu"
        )
        self.model = model.to(self.device).eval()
        self.scheduler = scheduler or DDIMScheduler()
        self.text_encoder = text_encoder
        self.vae_decoder = vae_decoder

    @torch.no_grad()
    def __call__(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        num_frames: int = 16,
        height: int = 256,
        width: int = 256,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate video from text.

        Returns:
            Video tensor ``(T, H, W, 3)`` with values in ``[0, 255]`` uint8.
        """
        if seed is not None:
            torch.manual_seed(seed)

        latent_h, latent_w = height // 8, width // 8
        latent_c = getattr(self.model, "in_channels", 4)
        latents = torch.randn(1, latent_c, num_frames, latent_h, latent_w, device=self.device)

        if context is None and self.text_encoder is not None:
            context = self._encode_text(prompt)
        elif context is None:
            context = torch.randn(1, 77, 768, device=self.device)

        neg_context = None
        if guidance_scale > 1.0:
            if self.text_encoder is not None and negative_prompt:
                neg_context = self._encode_text(negative_prompt)
            else:
                neg_context = torch.zeros_like(context)

        self.scheduler.set_timesteps(num_steps)

        for t in tqdm(self.scheduler.timesteps, desc="Generating video"):
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.long)

            noise_pred = self.model(latents, t_tensor, context=context)

            if guidance_scale > 1.0 and neg_context is not None:
                noise_uncond = self.model(latents, t_tensor, context=neg_context)
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            latents = self.scheduler.step(noise_pred, t.item(), latents)

        if self.vae_decoder is not None:
            video = self.vae_decoder(latents)
        else:
            video = latents

        video = self._latents_to_frames(video)
        return video

    def _encode_text(self, text: str) -> torch.Tensor:
        return self.text_encoder(text).to(self.device)

    @staticmethod
    def _latents_to_frames(latents: torch.Tensor) -> torch.Tensor:
        """Convert latent tensor to uint8 video frames ``(T, H, W, 3)``."""
        video = latents.squeeze(0)
        if video.shape[0] <= 4:
            video = video[:3]
        video = video.permute(1, 2, 3, 0)
        video = (video - video.min()) / (video.max() - video.min() + 1e-8)
        video = (video * 255).clamp(0, 255).to(torch.uint8)
        return video.cpu()

    def save_video(self, frames: torch.Tensor, path: str, fps: int = 8) -> str:
        """Save generated frames to an mp4 file using OpenCV."""
        import cv2

        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frames_np = frames.numpy()
        h, w = frames_np.shape[1], frames_np.shape[2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frames_np:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Video saved → {path} ({len(frames_np)} frames @ {fps}fps)")
        return path
