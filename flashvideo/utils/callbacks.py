"""Training callbacks for FlashVideo."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class Callback:
    """Base callback — override any hook method."""

    def on_train_start(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        pass


class CallbackRunner:
    """Execute a list of callbacks in order."""

    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        self.callbacks = callbacks or []

    def on_train_start(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_start(trainer)

    def on_train_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(trainer, epoch)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, metrics)

    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        for cb in self.callbacks:
            cb.on_step_end(trainer, step, loss)
