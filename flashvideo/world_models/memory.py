"""Long-horizon memory for world models.

Provides a memory bank that stores compressed historical context
beyond the transformer's window, enabling long-horizon simulation
without quadratic attention cost.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class CompressedMemoryBank(nn.Module):
    """Maintains a fixed-size compressed memory of past observations.

    New observations are compressed and added; old entries are evicted
    or further compressed using a learned aggregation.

    Args:
        mem_size: Maximum number of memory slots.
        dim: Feature dimension per slot.
        compress_ratio: Compression factor when evicting.
    """

    def __init__(self, mem_size: int = 128, dim: int = 768, compress_ratio: int = 2) -> None:
        super().__init__()
        self.mem_size = mem_size
        self.dim = dim
        self.compress_ratio = compress_ratio

        self.compressor = nn.Sequential(
            nn.Linear(dim * compress_ratio, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def compress(self, entries: torch.Tensor) -> torch.Tensor:
        """Compress *compress_ratio* entries into one."""
        b, n, d = entries.shape
        groups = n // self.compress_ratio
        remainder = n % self.compress_ratio

        if groups > 0:
            grouped = entries[:, :groups * self.compress_ratio].reshape(b, groups, self.compress_ratio * d)
            compressed = self.compressor(grouped)
        else:
            compressed = entries[:, :0]

        if remainder > 0:
            leftover = entries[:, groups * self.compress_ratio:]
            compressed = torch.cat([compressed, leftover], dim=1)

        return compressed

    def update(self, memory: torch.Tensor, new_entries: torch.Tensor) -> torch.Tensor:
        """Add *new_entries* to *memory*, compressing if over capacity.

        Args:
            memory: Current memory ``(B, M, D)``.
            new_entries: New observations ``(B, N, D)``.

        Returns:
            Updated memory ``(B, M', D)`` with ``M' <= mem_size``.
        """
        combined = torch.cat([memory, new_entries], dim=1)

        while combined.shape[1] > self.mem_size:
            excess = combined.shape[1] - self.mem_size
            to_compress = min(self.compress_ratio * ((excess // self.compress_ratio) + 1), combined.shape[1])
            head = combined[:, :to_compress]
            tail = combined[:, to_compress:]
            head = self.compress(head)
            combined = torch.cat([head, tail], dim=1)

        return combined


class LongHorizonMemory(nn.Module):
    """Memory-augmented module for long-horizon world model predictions.

    Wraps a ``CompressedMemoryBank`` with cross-attention so the world
    model's current hidden state can attend to compressed history.

    Args:
        dim: Feature dimension.
        mem_size: Memory bank capacity.
        num_heads: Cross-attention heads.
    """

    def __init__(self, dim: int = 768, mem_size: int = 128, num_heads: int = 8) -> None:
        super().__init__()
        self.memory_bank = CompressedMemoryBank(mem_size=mem_size, dim=dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_m = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: Current hidden states ``(B, T, D)``.
            memory: Memory bank ``(B, M, D)``.

        Returns:
            ``(augmented_hidden, updated_memory)``
        """
        q = self.norm_q(hidden)
        m = self.norm_m(memory)
        recalled, _ = self.cross_attn(q, m, m)
        gate = self.gate(recalled)
        augmented = hidden + gate * recalled

        updated_memory = self.memory_bank.update(memory, hidden.detach())

        return augmented, updated_memory

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create an empty memory tensor."""
        return torch.zeros(batch_size, 0, self.cross_attn.embed_dim, device=device)
