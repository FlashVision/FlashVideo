# Video Generation

FlashVideo supports three video generation modalities:

## Text-to-Video

Generate video from a text description using the **Video DiT** (Diffusion Transformer).

```python
from flashvideo.generation.text_to_video import TextToVideoPipeline
from flashvideo.models.architectures.video_dit import VideoDiT
from flashvideo.generation.schedulers import DDIMScheduler

model = VideoDiT(hidden_size=768, depth=12, num_heads=12)
pipeline = TextToVideoPipeline(model=model, scheduler=DDIMScheduler())

frames = pipeline(
    prompt="a mountain landscape at sunrise",
    num_frames=16,
    num_steps=50,
    guidance_scale=7.5,
)
pipeline.save_video(frames, "landscape.mp4", fps=8)
```

## Image-to-Video

Animate a static image:

```python
from flashvideo.generation.img_to_video import ImageToVideoPipeline

pipeline = ImageToVideoPipeline(model=model)
frames = pipeline(image=image_tensor, num_frames=16, motion_strength=1.0)
```

## Video Editing

Edit specific regions using mask-guided diffusion:

```python
from flashvideo.generation.video_editing import VideoEditingPipeline

pipeline = VideoEditingPipeline(model=model)
edited = pipeline(video=video_tensor, mask=mask_tensor, prompt="blue sky")
```

## Schedulers

| Scheduler | Steps | Quality | Speed | Use Case |
|-----------|-------|---------|-------|----------|
| DDPM | 1000 | Highest | Slowest | Reference quality |
| DDIM | 20-50 | High | Fast | Default choice |
| DPM-Solver++ | 15-25 | Highest | Fastest | Production |

## Video DiT Architecture

The Video Diffusion Transformer operates on 3D-patchified latent video tokens:

1. **3D Patch Embedding** — Converts `(B, C, T, H, W)` latents into flattened tokens
2. **Sinusoidal Timestep Embedding** — Conditions on diffusion step
3. **DiT Blocks** — Adaptive LayerNorm + joint space-time self-attention + cross-attention (text)
4. **Unpatchify** — Reconstruct spatial-temporal output

Key features:
- Adaptive layer norm (adaLN-Zero) for timestep conditioning
- Optional cross-attention for text guidance
- Classifier-free guidance at inference
