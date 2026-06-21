"""Training engine for FlashVideo models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from flashvideo.cfg.config import FlashVideoConfig
from flashvideo.utils.callbacks import CallbackRunner


class Trainer:
    """Unified trainer for video generation and understanding tasks.

    Handles mixed-precision training, gradient accumulation, checkpointing,
    and learning rate scheduling.
    """

    def __init__(
        self,
        cfg: Optional[FlashVideoConfig] = None,
        model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> None:
        self.cfg = cfg or FlashVideoConfig()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.callback_runner = CallbackRunner(callbacks or [])
        self.scaler = GradScaler(enabled=self.cfg.train.mixed_precision)
        self.device = self._resolve_device(self.cfg.device)
        self.global_step = 0
        self.best_loss = float("inf")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
        )

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.train.epochs)

    def train(self) -> Dict[str, float]:
        """Run the full training loop and return final metrics."""
        if self.model is None:
            raise RuntimeError("No model set. Pass a model to Trainer or build one from config.")
        if self.train_loader is None:
            raise RuntimeError("No training data loader set.")

        self.model = self.model.to(self.device)
        if self.optimizer is None:
            self.optimizer = self._build_optimizer()
        lr_scheduler = self._build_scheduler(self.optimizer)
        output_dir = Path(self.cfg.train.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.callback_runner.on_train_start(self)

        for epoch in range(1, self.cfg.train.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            self.callback_runner.on_epoch_start(self, epoch)
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg.train.epochs}")

            for step, batch in enumerate(pbar):
                loss = self._train_step(batch)
                epoch_loss += loss
                self.global_step += 1
                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

            avg_loss = epoch_loss / max(len(self.train_loader), 1)
            lr_scheduler.step()

            if self.val_loader is not None and epoch % self.cfg.train.eval_every == 0:
                val_metrics = self._validate()
                print(f"  Validation: {val_metrics}")

            if epoch % self.cfg.train.save_every == 0:
                self._save_checkpoint(output_dir / f"checkpoint_epoch{epoch}.pth", epoch)

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self._save_checkpoint(output_dir / "best.pth", epoch)

            self.callback_runner.on_epoch_end(self, epoch, {"loss": avg_loss})

        self.callback_runner.on_train_end(self)
        return {"best_loss": self.best_loss, "final_step": self.global_step}

    def _train_step(self, batch: Dict[str, Any]) -> float:
        video = batch["video"].to(self.device)
        self.optimizer.zero_grad()

        with autocast(enabled=self.cfg.train.mixed_precision):
            output = self.model(video)
            loss = output if isinstance(output, torch.Tensor) and output.ndim == 0 else output.get("loss", output)

        self.scaler.scale(loss).backward()

        if self.cfg.train.max_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                video = batch["video"].to(self.device)
                output = self.model(video)
                loss = output if isinstance(output, torch.Tensor) and output.ndim == 0 else output.get("loss", output)
                total_loss += loss.item()
                count += 1
        return {"val_loss": total_loss / max(count, 1)}

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            },
            path,
        )
        print(f"  Checkpoint saved: {path}")
