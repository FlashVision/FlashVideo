<div align="center">

# FlashVideo

**Video Understanding & Generation — Text-to-Video, Action Recognition, World Models, and Temporal AI**

[![PyPI](https://img.shields.io/badge/PyPI-flashvideo-blue.svg)](https://pypi.org/project/flashvideo/)
[![CI](https://github.com/FlashVision/FlashVideo/actions/workflows/ci.yml/badge.svg)](https://github.com/FlashVision/FlashVideo/actions)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![ONNX](https://img.shields.io/badge/ONNX-supported-005CED.svg)](https://onnx.ai)
[![LoRA](https://img.shields.io/badge/LoRA-supported-brightgreen.svg)](#lora-fine-tuning)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Documentation](docs/Home.md) · [Quick Start](docs/Quick-Start.md) · [Generation](docs/Video-Generation.md) · [Understanding](docs/Video-Understanding.md) · [Action Recognition](docs/Action-Recognition.md) · [World Models](docs/World-Models.md)

</div>

---

## What is FlashVideo?

FlashVideo is a modular, production-ready framework for **video AI** — covering generation, understanding, and simulation. It provides everything you need to **generate videos from text**, **recognize actions**, **caption video content**, and **simulate environments** with physics-aware world models.

### Key Features

- **Video DiT** — Sora/Veo-style Diffusion Transformer for text-to-video generation
- **Video ViT** — Tubelet-embedding Vision Transformer for video understanding
- **TimeSformer** — Divided space-time attention for efficient action recognition
- **World Models** — NVIDIA Cosmos-inspired autoregressive environment simulation
- **3 Schedulers** — DDPM, DDIM, DPM-Solver++ adapted for temporal diffusion
- **LoRA Fine-Tuning** — Parameter-efficient adaptation for all architectures
- **Physics-Aware Dynamics** — Energy conservation priors for world models
- **Long-Horizon Memory** — Compressed memory bank for extended simulations
- **ONNX Export** — Deploy with ONNX Runtime or TensorRT
- **Registry System** — Plug in custom models, schedulers, and datasets via config
- **Mixed Precision** — AMP training with automatic loss scaling
- **CLI & Python API** — Both command-line and programmatic interfaces

---

## Installation

```bash
pip install flashvideo
```

**With extras:**

```bash
pip install flashvideo[analytics]    # matplotlib, pandas
pip install flashvideo[export]       # onnx, onnxruntime
pip install flashvideo[quality]      # torch-fidelity for FVD/FID
pip install flashvideo[all]          # everything
```

**From source:**

```bash
git clone https://github.com/FlashVision/FlashVideo.git
cd FlashVideo
pip install -e ".[dev,all]"
```

**Verify:**

```bash
flashvideo check
```

---

## Usage

### Python API

```python
from flashvideo import VideoGenerator, ActionClassifier, SceneSimulator

# Generate a video
gen = VideoGenerator()
gen.generate("a sunset over the ocean", output="sunset.mp4")

# Classify actions
clf = ActionClassifier()
results = clf.classify("video.mp4", top_k=5)

# Simulate a scene
sim = SceneSimulator()
sim.simulate("a robot arm picking up a block", output="sim.mp4")
```

### CLI

```bash
# Generate from text
flashvideo generate --prompt "a cat in a garden" --frames 16 --steps 50

# Classify actions
flashvideo classify --video input.mp4 --top-k 5

# World model simulation
flashvideo simulate --prompt "a ball bouncing" --frames 32

# Train a model
flashvideo train --config configs/flashvideo_generation.yaml

# Export to ONNX
flashvideo export --model model.pth --output model.onnx

# Benchmark performance
flashvideo benchmark --device cuda --resolution 256
```

---

## Models

| Model | Type | Parameters | Description |
|-------|------|------------|-------------|
| Video DiT | Generation | ~85M (base) | Diffusion Transformer for text-to-video |
| Video ViT | Understanding | ~86M (base) | Tubelet-embedding ViT for classification |
| TimeSformer | Recognition | ~86M (base) | Divided space-time attention |
| World Model | Simulation | ~90M (base) | Causal transformer for dynamics prediction |

---

## Schedulers

| Scheduler | Steps | Quality | Speed | Use Case |
|-----------|-------|---------|-------|----------|
| DDPM | 1000 | ★★★★★ | ★☆☆☆☆ | Reference quality |
| DDIM | 20-50 | ★★★★☆ | ★★★★☆ | Fast, deterministic |
| DPM-Solver++ | 15-25 | ★★★★★ | ★★★★★ | Best quality/speed |

---

## Solutions

| Solution | Description |
|----------|-------------|
| `VideoGenerator` | High-level text/image → video generation |
| `ActionClassifier` | Video action recognition with top-K predictions |
| `SceneSimulator` | World model-based environment simulation |

---

## Training

### Standard Training

```bash
flashvideo train --config configs/flashvideo_generation.yaml
```

### LoRA Fine-Tuning

```python
from flashvideo.models.lora import apply_lora
model = apply_lora(model, rank=8, alpha=16)
# Trainable: ~1-2% of parameters
```

### World Model Training

```bash
flashvideo train --config configs/flashvideo_world_model.yaml
```

---

## Analytics

```bash
flashvideo benchmark --device cuda --frames 16 --resolution 256 --iterations 100
```

Reports:
- Average inference time (ms)
- Throughput (videos/sec)
- Peak GPU memory (MB)
- Model parameter count

### Metrics

| Metric | Task | Description |
|--------|------|-------------|
| FVD | Generation | Fréchet Video Distance |
| FID | Generation | Fréchet Inception Distance |
| IS | Generation | Inception Score |
| Temporal Consistency | Generation | Frame-to-frame smoothness |
| Top-1/5 Accuracy | Recognition | Classification accuracy |

---

## Examples

| Example | Script | Description |
|---------|--------|-------------|
| Text-to-Video | `examples/text_to_video.py` | Generate video from text |
| Understanding | `examples/video_understanding.py` | Classification + event detection |
| Action Recognition | `examples/action_recognition.py` | TimeSformer action classification |
| World Model | `examples/world_model.py` | Physics-aware simulation |
| Benchmark | `examples/benchmark_video.py` | Performance benchmarks |

```bash
# Quick demo (runs on CPU, no pretrained weights needed)
python examples/text_to_video.py --prompt "a mountain landscape" --steps 10
```

---

## Project Structure

```
FlashVideo/
├── flashvideo/                     # Core library
│   ├── models/                     # Model definitions
│   │   ├── architectures/          # Video DiT, Video ViT, TimeSformer, World Model
│   │   ├── flashvideo_model.py     # Unified model wrapper
│   │   └── lora.py                 # LoRA adaptation
│   ├── generation/                 # Text-to-video, img-to-video, editing, schedulers
│   ├── understanding/              # Classification, captioning, temporal, grounding
│   ├── world_models/               # Simulator, dynamics, action-conditioned, memory
│   ├── engine/                     # Trainer, Validator, Predictor, Exporter
│   ├── data/                       # Datasets, transforms, video reader, samplers
│   ├── solutions/                  # VideoGenerator, ActionClassifier, SceneSimulator
│   ├── analytics/                  # Benchmark, FVD/FID/IS metrics
│   ├── utils/                      # I/O, visualization, callbacks
│   ├── cfg/                        # Dataclass configuration
│   ├── registry.py                 # Pluggable component registry
│   └── cli.py                      # Command-line interface
├── configs/                        # YAML configuration files
├── examples/                       # Runnable example scripts
├── tests/                          # Pytest test suite
├── docs/                           # Documentation
├── docker/                         # Dockerfile & docker-compose
├── .github/workflows/ci.yml        # GitHub Actions CI
├── pyproject.toml                  # Build configuration
├── CONTRIBUTING.md                 # Contribution guide
├── CHANGELOG.md                    # Version history
├── LICENSE                         # MIT License
└── README.md                       # This file
```

---

## Docker

```bash
# Build
docker build -t flashvideo -f docker/Dockerfile .

# Run with GPU
docker run --gpus all -it flashvideo check

# Docker Compose
docker compose -f docker/docker-compose.yml up
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/FlashVision/FlashVideo.git
cd FlashVideo
pip install -e ".[dev,all]"
pytest tests/ -v
ruff check flashvideo/
```

---

## License

FlashVideo is released under the [MIT License](LICENSE).

---

<div align="center">
  <b>Part of the <a href="https://github.com/FlashVision">FlashVision</a> ecosystem</b>
</div>
