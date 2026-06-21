# Action Recognition

FlashVideo implements state-of-the-art action recognition using the **TimeSformer** architecture with divided space-time attention.

## Quick Start

```python
from flashvideo import ActionClassifier

clf = ActionClassifier(num_classes=400)
results = clf.classify("basketball.mp4", top_k=5)
for label, score in results:
    print(f"{score:.4f}  {label}")
```

## TimeSformer Architecture

TimeSformer factorises the full space-time self-attention into two efficient passes:

### Divided Space-Time Attention

Each transformer block performs:

1. **Temporal Attention** — Each spatial patch attends to the same patch across all frames
2. **Spatial Attention** — Each frame's patches attend to each other (within-frame)
3. **MLP** — Standard feed-forward network

This factorisation reduces complexity from `O((T*S)^2)` to `O(T^2*S + T*S^2)`.

## Training

```bash
flashvideo train --config configs/flashvideo_action_recognition.yaml
```

### Configuration

```yaml
model:
  name: TimeSformer
  embed_dim: 768
  depth: 12
  num_heads: 12
  patch_size: 16
  num_frames: 8
  image_size: 224

data:
  dataset: kinetics
  root: data/kinetics400/
  num_frames: 8
  frame_stride: 4
```

## LoRA Fine-Tuning

Efficiently adapt a pre-trained TimeSformer:

```python
from flashvideo.models.architectures.timesformer import TimeSformer
from flashvideo.models.lora import apply_lora

model = TimeSformer(num_classes=400)
model = apply_lora(model, rank=8, alpha=16)
# Only ~1-2% of parameters are trainable
```

## Supported Datasets

- **Kinetics-400/600/700** — Large-scale action recognition
- **Something-Something V2** — Fine-grained temporal reasoning
- **Custom datasets** — Any folder of videos with class subdirectories
