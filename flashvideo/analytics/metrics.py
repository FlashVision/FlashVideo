"""Video generation and understanding quality metrics.

Implements or wraps standard metrics: FVD (Fréchet Video Distance),
FID (Fréchet Inception Distance), IS (Inception Score), accuracy,
and temporal consistency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def compute_fvd(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
) -> float:
    """Fréchet Video Distance between two sets of video features.

    Both inputs should be ``(N, D)`` feature matrices extracted from a
    pre-trained video encoder (e.g. I3D).
    """
    mu_r = real_features.mean(dim=0)
    mu_g = generated_features.mean(dim=0)

    sigma_r = _cov(real_features)
    sigma_g = _cov(generated_features)

    diff = mu_r - mu_g
    covmean = _sqrtm_newton(sigma_r @ sigma_g)

    fvd = diff.dot(diff) + torch.trace(sigma_r + sigma_g - 2 * covmean)
    return fvd.item()


def compute_fid(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
) -> float:
    """Fréchet Inception Distance (image-level)."""
    return compute_fvd(real_features, generated_features)


def compute_inception_score(
    logits: torch.Tensor,
    num_splits: int = 10,
) -> tuple:
    """Inception Score from classifier logits.

    Returns:
        ``(mean_is, std_is)``
    """
    probs = F.softmax(logits, dim=-1)
    scores = []
    chunk_size = len(probs) // num_splits

    for i in range(num_splits):
        chunk = probs[i * chunk_size : (i + 1) * chunk_size]
        if len(chunk) == 0:
            continue
        marginal = chunk.mean(dim=0, keepdim=True)
        kl = chunk * (chunk.log() - marginal.log())
        scores.append(kl.sum(dim=-1).mean().exp().item())

    return float(np.mean(scores)), float(np.std(scores))


def compute_temporal_consistency(
    video: torch.Tensor,
    metric: str = "mse",
) -> float:
    """Measure temporal consistency of a video.

    Lower values indicate smoother, more consistent video.

    Args:
        video: ``(T, C, H, W)`` or ``(B, C, T, H, W)`` video tensor.
        metric: ``"mse"`` or ``"cosine"``.
    """
    if video.ndim == 5:
        video = video.squeeze(0).permute(1, 0, 2, 3)

    if video.shape[0] < 2:
        return 0.0

    if metric == "cosine":
        flat = video.flatten(1)
        sims = F.cosine_similarity(flat[:-1], flat[1:], dim=-1)
        return 1.0 - sims.mean().item()

    diffs = (video[1:] - video[:-1]).pow(2).mean()
    return diffs.item()


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Top-1 accuracy."""
    preds = predictions.argmax(dim=-1) if predictions.ndim > 1 else predictions
    return (preds == targets).float().mean().item()


def compute_top_k_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> float:
    """Top-K accuracy."""
    _, topk_preds = predictions.topk(k, dim=-1)
    correct = topk_preds.eq(targets.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _cov(x: torch.Tensor) -> torch.Tensor:
    """Compute covariance matrix."""
    n = x.shape[0]
    mean = x.mean(dim=0, keepdim=True)
    x_centered = x - mean
    return (x_centered.T @ x_centered) / max(n - 1, 1)


def _sqrtm_newton(m: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """Approximate matrix square root via Newton-Schulz iteration."""
    dim = m.shape[0]
    norm = m.norm()
    if norm < 1e-8:
        return torch.zeros_like(m)

    y = m / norm
    z = torch.eye(dim, device=m.device, dtype=m.dtype)

    for _ in range(num_iters):
        t = 0.5 * (3.0 * torch.eye(dim, device=m.device, dtype=m.dtype) - z @ y)
        y = y @ t
        z = t @ z

    return y * norm.sqrt()
