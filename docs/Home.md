# FlashVideo Documentation

Welcome to the FlashVideo documentation! FlashVideo is a production-ready framework for **video understanding and generation**, covering text-to-video generation, action recognition, video captioning, and world model simulation.

## Contents

| Page | Description |
|------|-------------|
| [Installation](Installation.md) | Install FlashVideo and dependencies |
| [Quick Start](Quick-Start.md) | Get running in 5 minutes |
| [Video Generation](Video-Generation.md) | Text-to-video, image-to-video, video editing |
| [Video Understanding](Video-Understanding.md) | Classification, captioning, temporal grounding |
| [Action Recognition](Action-Recognition.md) | TimeSformer, Video ViT for action classification |
| [World Models](World-Models.md) | Physics-aware simulation, action-conditioned generation |
| [FAQ](FAQ.md) | Frequently asked questions |

## Key Concepts

### Video Generation
FlashVideo uses a **Video Diffusion Transformer (Video DiT)** with 3D patchification and joint space-time attention to generate temporally-consistent video from text prompts. Supports DDPM, DDIM, and DPM-Solver++ schedulers.

### Video Understanding
**Video ViT** with tubelet embeddings processes entire video clips for classification, captioning, and temporal analysis. **TimeSformer** factorises attention into separate temporal and spatial passes for efficient long-range temporal modeling.

### World Models
Inspired by NVIDIA Cosmos and GAIA-1, the **World Model Transformer** autoregressively predicts future latent frames conditioned on actions. A physics-aware dynamics model enforces energy conservation for more realistic simulations.

## Architecture

```
FlashVideo
├── Generation    → Video DiT + Schedulers → Text/Image → Video
├── Understanding → Video ViT / TimeSformer → Classification, Captioning
├── World Models  → Causal Transformer + Dynamics → Simulation
└── Solutions     → VideoGenerator, ActionClassifier, SceneSimulator
```
