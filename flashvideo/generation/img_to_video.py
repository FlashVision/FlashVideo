"""Image-to-video animation pipeline.

Takes a single image and generates a video sequence by conditioning the
diffusion process on the image's latent representation as the first frame.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from flashvideo.generation.schedulers import BaseScheduler, DDIMScheduler


class ImageToVideoPipeline:
    """Animate a static image into a video clip.

    The first frame is encoded to a latent and used as a conditioning
    signal.  Remaining frames are generated via iterative denoising.
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

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a ``(1, 3, H, W)`` image to latent space."""
        if self.vae_encoder is not None:
            return self.vae_encoder(image)
        return F.interpolate(image, scale_factor=0.125, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def __call__(
        self,
        image: torch.Tensor,
        num_frames: int = 16,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        motion_strength: float = 1.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate video from an image.

        Args:
            image: Input image ``(1, 3, H, W)`` float in ``[0, 1]``.
            num_frames: Number of output frames.
            num_steps: Diffusion denoising steps.
            guidance_scale: Classifier-free guidance weight.
            motion_strength: Scale of initial noise (higher = more motion).
            seed: Random seed for reproducibility.

        Returns:
            Video ``(T, H, W, 3)`` uint8.
        """
        if seed is not None:
            torch.manual_seed(seed)

        image = image.to(self.device)
        img_latent = self._encode_image(image)
        _, c, h, w = img_latent.shape

        noise = torch.randn(1, c, num_frames, h, w, device=self.device) * motion_strength
        noise[:, :, 0] = 0
        latents = img_latent.unsqueeze(2).expand(-1, -1, num_frames, -1, -1) + noise

        self.scheduler.set_timesteps(num_steps)
        for t in tqdm(self.scheduler.timesteps, desc="Animating"):
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.long)
            noise_pred = self.model(latents, t_tensor)
            latents = self.scheduler.step(noise_pred, t.item(), latents)
            latents[:, :, 0] = img_latent.squeeze(2)

        if self.vae_decoder is not None:
            video = self.vae_decoder(latents)
        else:
            video = latents

        video = video.squeeze(0)[:3].permute(1, 2, 3, 0)
        video = (video - video.min()) / (video.max() - video.min() + 1e-8)
        return (video * 255).clamp(0, 255).to(torch.uint8).cpu()
