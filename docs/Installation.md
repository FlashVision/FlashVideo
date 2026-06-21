# Installation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (recommended for GPU acceleration)

## Quick Install

```bash
pip install flashvideo
```

## With Extras

```bash
pip install flashvideo[analytics]    # matplotlib, pandas
pip install flashvideo[export]       # ONNX export
pip install flashvideo[quality]      # torch-fidelity for FVD/FID
pip install flashvideo[all]          # everything
```

## From Source

```bash
git clone https://github.com/FlashVision/FlashVideo.git
cd FlashVideo
pip install -e ".[dev,all]"
```

## One-Command Setup

For a fully configured environment (auto-detects GPU):

```bash
bash setup_env.sh
source venv/bin/activate
```

Force CPU or specific CUDA version:

```bash
bash setup_env.sh --cpu
bash setup_env.sh --cuda 12.4
```

## Verify Installation

```bash
flashvideo check
flashvideo settings
```

## Docker

```bash
docker build -t flashvideo -f docker/Dockerfile .
docker run --gpus all -it flashvideo check
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0.0 | Deep learning framework |
| torchvision | ≥0.15.0 | Vision utilities |
| transformers | ≥4.30.0 | Text encoders |
| diffusers | ≥0.25.0 | Diffusion pipeline components |
| numpy | ≥1.24.0 | Numerical computing |
| opencv-python | ≥4.8.0 | Video I/O |
| Pillow | ≥9.0.0 | Image processing |
| PyYAML | ≥6.0 | Configuration |
| tqdm | ≥4.65.0 | Progress bars |
| decord | ≥0.6.0 | GPU-accelerated video decoding |
| safetensors | ≥0.4.0 | Safe model serialisation |
