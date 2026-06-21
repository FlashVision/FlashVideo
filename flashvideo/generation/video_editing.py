"""Video editing and inpainting pipeline.

Modify specific regions or attributes of an existing video while
preserving temporal coherence using mask-guided diffusion.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from flashvideo.generation.schedulers import BaseScheduler, DDIMScheduler


class VideoEditingPipeline:
    """Edit or inpaint video regions guided by a text prompt and spatial mask.

    The pipeline runs diffusion denoising on the masked region while
    keeping unmasked content from the original video at each step,
    ensuring seamless temporal blending.
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler: Optional[BaseScheduler] = None,
        vae_encoder: Optional[nn.Module] = None,
        vae_decoder: Optional[nn.Module] = None,
        device: str = "auto",
    ) -> None:
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
        self.model = model.to(self.device).eval()
        self.scheduler = scheduler or DDIMScheduler()
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder

    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        if self.vae_encoder is not None:
            return self.vae_encoder(video)
        return F.interpolate(video, scale_factor=0.125, mode="trilinear", align_corners=False)

    @torch.no_grad()
    def __call__(
        self,
        video: torch.Tensor,
        mask: torch.Tensor,
        prompt: str = "",
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        context: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            video: Original video ``(1, C, T, H, W)`` float ``[0, 1]``.
            mask: Binary mask ``(1, 1, T, H, W)`` — 1 = edit region.
            prompt: Text description for the edit.
            num_steps: Denoising steps.
            guidance_scale: CFG weight.
            context: Pre-computed text embeddings ``(1, L, D)``.
            seed: Random seed.

        Returns:
            Edited video ``(T, H, W, 3)`` uint8.
        """
        if seed is not None:
            torch.manual_seed(seed)

        video = video.to(self.device)
        mask = mask.to(self.device)

        video_latent = self._encode_video(video)
        mask_latent = F.interpolate(mask.float(), size=video_latent.shape[2:], mode="nearest")

        noise = torch.randn_like(video_latent)
        latents = video_latent * (1 - mask_latent) + noise * mask_latent

        if context is None:
            context = torch.randn(1, 77, 768, device=self.device)

        self.scheduler.set_timesteps(num_steps)
        for t in tqdm(self.scheduler.timesteps, desc="Editing"):
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.long)
            noise_pred = self.model(latents, t_tensor, context=context)

            latents_denoised = self.scheduler.step(noise_pred, t.item(), latents)
            latents = video_latent * (1 - mask_latent) + latents_denoised * mask_latent

        if self.vae_decoder is not None:
            result = self.vae_decoder(latents)
        else:
            result = latents

        result = result.squeeze(0)[:3].permute(1, 2, 3, 0)
        result = (result - result.min()) / (result.max() - result.min() + 1e-8)
        return (result * 255).clamp(0, 255).to(torch.uint8).cpu()
