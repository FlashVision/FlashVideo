"""LoRA — Low-Rank Adaptation for parameter-efficient fine-tuning.

Supports standard LoRA injection into any ``nn.Linear`` layer in a
FlashVideo model.  Freeze the base weights and train only the small
rank-decomposition matrices.
"""

from __future__ import annotations

import math
from typing import List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A drop-in replacement for ``nn.Linear`` with LoRA adapters.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        bias: Whether to include bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
        return base + lora

    def merge(self) -> nn.Linear:
        """Return a standard ``nn.Linear`` with LoRA weights merged in."""
        merged = nn.Linear(self.linear.in_features, self.linear.out_features, bias=self.linear.bias is not None)
        merged.weight.data = self.linear.weight.data + (self.lora_b @ self.lora_a) * self.scaling
        if self.linear.bias is not None:
            merged.bias.data = self.linear.bias.data
        return merged


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[Set[str]] = None,
) -> nn.Module:
    """Replace matching ``nn.Linear`` layers with ``LoRALinear`` in-place.

    Args:
        model: The model to inject LoRA into.
        rank: LoRA rank.
        alpha: Scaling factor.
        target_modules: Set of module name substrings to match.
            Defaults to ``{"qkv", "proj", "linear"}`` to target attention layers.
    """
    if target_modules is None:
        target_modules = {"qkv", "proj", "linear"}

    replaced: List[str] = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue

        parts = name.rsplit(".", 1)
        parent = model if len(parts) == 1 else dict(model.named_modules())[parts[0]]
        attr = parts[-1]

        lora_layer = LoRALinear(
            module.in_features,
            module.out_features,
            rank=rank,
            alpha=alpha,
            bias=module.bias is not None,
        )
        lora_layer.linear.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            lora_layer.linear.bias.data.copy_(module.bias.data)

        setattr(parent, attr, lora_layer)
        replaced.append(name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA applied to {len(replaced)} layers | Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )
    return model
