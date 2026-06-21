"""Dataclass-based configuration for FlashVideo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import yaml


@dataclass
class ModelConfig:
    name: str = "VideoDiT"
    in_channels: int = 4
    out_channels: int = 4
    hidden_size: int = 768
    num_heads: int = 12
    depth: int = 12
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    num_frames: int = 16
    image_size: int = 256
    pretrained: Optional[str] = None


@dataclass
class DataConfig:
    dataset: str = "folder"
    root: str = "data/"
    num_frames: int = 16
    frame_stride: int = 1
    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 4
    sampler: str = "uniform"


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    scheduler: str = "cosine"
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    save_every: int = 5
    eval_every: int = 5
    output_dir: str = "workspace/"
    resume: Optional[str] = None


@dataclass
class GenerationConfig:
    scheduler: str = "DDIM"
    num_steps: int = 50
    guidance_scale: float = 7.5
    num_frames: int = 16
    fps: int = 8
    eta: float = 0.0


@dataclass
class FlashVideoConfig:
    task: str = "text_to_video"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    device: str = "auto"
    seed: int = 42


def _dict_to_dataclass(dc_cls, d: dict):
    """Recursively convert a dict to a dataclass, ignoring unknown keys."""
    import dataclasses

    if not dataclasses.is_dataclass(dc_cls):
        return d

    fieldtypes = {f.name: f.type for f in dataclasses.fields(dc_cls)}
    kwargs = {}
    for k, v in d.items():
        if k in fieldtypes:
            ft = fieldtypes[k]
            if isinstance(ft, str):
                ft = eval(ft)  # noqa: S307
            if dataclasses.is_dataclass(ft) and isinstance(v, dict):
                kwargs[k] = _dict_to_dataclass(ft, v)
            else:
                kwargs[k] = v
    return dc_cls(**kwargs)


def load_config(path: str | Path) -> FlashVideoConfig:
    """Load a YAML config and return a ``FlashVideoConfig`` instance."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _dict_to_dataclass(FlashVideoConfig, raw)


def merge_cli_args(cfg: FlashVideoConfig, args: argparse.Namespace) -> FlashVideoConfig:
    """Override config fields with non-None CLI arguments."""
    import dataclasses

    for f in dataclasses.fields(cfg):
        cli_val = getattr(args, f.name, None)
        if cli_val is not None:
            setattr(cfg, f.name, cli_val)
    return cfg
