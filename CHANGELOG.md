# Changelog

All notable changes to FlashVideo will be documented in this file.

## [1.0.0] — 2026-06-21

### Added
- **Package structure** — `pip install` from GitHub or PyPI
- **CLI** — `flashvideo generate`, `train`, `classify`, `simulate`, `export`, `benchmark`, `check`, `settings`, `version`
- **Python API** — `FlashVideo`, `Trainer`, `Predictor`, `Validator`, `Exporter`
- **Video Generation** — Text-to-video, image-to-video, video editing/inpainting pipelines
- **Schedulers** — DDPM, DDIM, DPM-Solver++ adapted for temporal diffusion
- **Video Understanding** — Classification, captioning, temporal grounding, event detection
- **Action Recognition** — TimeSformer, Video ViT with Kinetics/Something-Something support
- **World Models** — NVIDIA Cosmos-style environment simulation, physics-aware dynamics
- **Models** — Video DiT (diffusion transformer), Video ViT, TimeSformer, world model architectures
- **LoRA training** — Parameter-efficient fine-tuning for video models
- **Solutions** — VideoGenerator, ActionClassifier, SceneSimulator high-level APIs
- **Analytics** — FVD, FID, IS, accuracy, temporal consistency benchmarks
- **Export** — ONNX export support
- **Mixed precision** — AMP (FP16) training with gradient checkpointing
- **CI/CD** — GitHub Actions (lint + test on Python 3.9-3.12)
- **Examples** — 5 runnable example scripts

### Architecture
- Video DiT with 3D attention (spatial + temporal) for generation
- Video ViT with tubelet embedding for understanding
- TimeSformer divided space-time attention for action recognition
- Physics-aware world model with action-conditioned dynamics
- Temporal consistency via cross-frame attention and optical flow regularization
