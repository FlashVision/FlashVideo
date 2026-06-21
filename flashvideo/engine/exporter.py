"""Model export to ONNX and other deployment formats."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


class Exporter:
    """Export a FlashVideo model to deployment-ready formats."""

    def export(
        self,
        model_or_path: nn.Module | str,
        output_path: str = "model.onnx",
        fmt: str = "onnx",
        num_frames: int = 16,
        image_size: int = 256,
        opset: int = 17,
    ) -> str:
        """Export *model_or_path* and return the path to the exported file."""
        if fmt == "onnx":
            return self._export_onnx(model_or_path, output_path, num_frames, image_size, opset)
        raise ValueError(f"Unsupported export format: {fmt}")

    def _export_onnx(
        self,
        model_or_path: nn.Module | str,
        output_path: str,
        num_frames: int,
        image_size: int,
        opset: int,
    ) -> str:
        if isinstance(model_or_path, str):
            state = torch.load(model_or_path, map_location="cpu", weights_only=True)
            raise RuntimeError(
                "Cannot reconstruct model architecture from a state dict alone. "
                "Pass an nn.Module instance instead, or a full checkpoint "
                "that includes model config."
            )

        model = model_or_path
        model.eval()
        dummy = torch.randn(1, 3, num_frames, image_size, image_size)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            dummy,
            output_path,
            opset_version=opset,
            input_names=["video"],
            output_names=["output"],
            dynamic_axes={
                "video": {0: "batch", 2: "frames"},
                "output": {0: "batch"},
            },
        )
        print(f"Exported ONNX model → {output_path}")
        return output_path
