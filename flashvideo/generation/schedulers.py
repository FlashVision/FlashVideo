"""Noise schedulers for video diffusion — DDPM, DDIM, DPM-Solver++.

All schedulers share a common interface so they can be swapped via config.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from flashvideo.registry import SCHEDULERS


class BaseScheduler:
    """Base noise scheduler with a linear beta schedule."""

    def __init__(
        self,
        num_train_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ) -> None:
        self.num_train_steps = num_train_steps
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_steps) ** 2
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.timesteps: Optional[torch.Tensor] = None

    def set_timesteps(self, num_steps: int) -> None:
        step_ratio = self.num_train_steps // num_steps
        self.timesteps = torch.arange(0, num_steps, device="cpu") * step_ratio
        self.timesteps = self.timesteps.flip(0)

    def add_noise(self, x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ac = self.alphas_cumprod.to(x.device)
        sqrt_ac = ac[t].sqrt().view(-1, *([1] * (x.ndim - 1)))
        sqrt_one_minus = (1 - ac[t]).sqrt().view(-1, *([1] * (x.ndim - 1)))
        return sqrt_ac * x + sqrt_one_minus * noise

    def step(self, model_output: torch.Tensor, t: int, sample: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError


@SCHEDULERS.register("DDPM")
class DDPMScheduler(BaseScheduler):
    """Denoising Diffusion Probabilistic Models scheduler."""

    def step(self, model_output: torch.Tensor, t: int, sample: torch.Tensor, **kwargs) -> torch.Tensor:
        ac = self.alphas_cumprod.to(sample.device)
        alpha_t = ac[t]
        alpha_prev = ac[t - 1] if t > 0 else torch.tensor(1.0, device=sample.device)
        beta_t = 1 - alpha_t / alpha_prev

        pred_x0 = (sample - (1 - alpha_t).sqrt() * model_output) / alpha_t.sqrt()
        pred_x0 = pred_x0.clamp(-1, 1)

        mean = alpha_prev.sqrt() * beta_t / (1 - alpha_t) * pred_x0
        mean = mean + (alpha_t / alpha_prev).sqrt() * (1 - alpha_prev) / (1 - alpha_t) * sample

        if t > 0:
            noise = torch.randn_like(sample)
            variance = beta_t * (1 - alpha_prev) / (1 - alpha_t)
            return mean + variance.sqrt() * noise
        return mean


@SCHEDULERS.register("DDIM")
class DDIMScheduler(BaseScheduler):
    """Denoising Diffusion Implicit Models scheduler — deterministic by default."""

    def step(
        self,
        model_output: torch.Tensor,
        t: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        ac = self.alphas_cumprod.to(sample.device)
        alpha_t = ac[t]
        alpha_prev = ac[t - 1] if t > 0 else torch.tensor(1.0, device=sample.device)

        pred_x0 = (sample - (1 - alpha_t).sqrt() * model_output) / alpha_t.sqrt()
        pred_x0 = pred_x0.clamp(-1, 1)

        sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
        dir_xt = (1 - alpha_prev - sigma**2).clamp(min=0).sqrt() * model_output

        x_prev = alpha_prev.sqrt() * pred_x0 + dir_xt
        if eta > 0 and t > 0:
            x_prev = x_prev + sigma * torch.randn_like(sample)
        return x_prev


@SCHEDULERS.register("DPM++")
class DPMPPScheduler(BaseScheduler):
    """DPM-Solver++ second-order scheduler for fast, high-quality sampling."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._prev_output: Optional[torch.Tensor] = None
        self._prev_t: Optional[int] = None

    def set_timesteps(self, num_steps: int) -> None:
        super().set_timesteps(num_steps)
        self._prev_output = None
        self._prev_t = None

    def _lambda(self, t: int) -> float:
        ac = self.alphas_cumprod[t].item()
        return 0.5 * math.log(ac / (1 - ac))

    def step(self, model_output: torch.Tensor, t: int, sample: torch.Tensor, **kwargs) -> torch.Tensor:
        ac = self.alphas_cumprod.to(sample.device)
        alpha_t = ac[t]
        pred_x0 = (sample - (1 - alpha_t).sqrt() * model_output) / alpha_t.sqrt()
        pred_x0 = pred_x0.clamp(-1, 1)

        t_prev = max(t - (self.num_train_steps // len(self.timesteps)), 0)
        alpha_prev = ac[t_prev]

        if self._prev_output is not None and self._prev_t is not None:
            lam_t = self._lambda(t)
            lam_prev = self._lambda(self._prev_t)
            lam_next = self._lambda(t_prev)

            h = lam_t - lam_prev
            r = (lam_next - lam_t) / (2 * h) if h != 0 else 0.5
            corrected = (1 + r) * pred_x0 - r * self._prev_output
            x_prev = alpha_prev.sqrt() * corrected + (1 - alpha_prev).sqrt() * model_output
        else:
            x_prev = alpha_prev.sqrt() * pred_x0 + (1 - alpha_prev).sqrt() * model_output

        self._prev_output = pred_x0
        self._prev_t = t
        return x_prev
