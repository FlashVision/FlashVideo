"""Validation engine for FlashVideo models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Validator:
    """Evaluate a video model on a validation dataset and report metrics."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "auto",
        metrics: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
        self.metric_names = metrics or ["loss", "accuracy"]

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run one full validation pass and return aggregated metrics."""
        self.model.eval()
        self.model.to(self.device)

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(self.dataloader, desc="Validating"):
            video = batch["video"].to(self.device)
            labels = batch.get("label")

            output = self.model(video)

            if isinstance(output, dict):
                loss = output.get("loss", torch.tensor(0.0))
                logits = output.get("logits")
            elif isinstance(output, torch.Tensor) and output.ndim == 0:
                loss = output
                logits = None
            else:
                loss = torch.tensor(0.0)
                logits = output

            total_loss += loss.item()

            if logits is not None and labels is not None:
                labels = labels.to(self.device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=self.device)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()

        n = max(len(self.dataloader), 1)
        results: Dict[str, float] = {"val_loss": total_loss / n}
        if total > 0:
            results["accuracy"] = correct / total
        return results
