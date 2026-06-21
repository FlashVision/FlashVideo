# Quick Start

Get running with FlashVideo in under 5 minutes.

## 1. Generate a Video

### Python API

```python
from flashvideo import VideoGenerator

gen = VideoGenerator()
gen.generate("a cat playing in a garden", output="cat.mp4", num_steps=20)
```

### CLI

```bash
flashvideo generate --prompt "a sunset over the ocean" --output sunset.mp4 --steps 20
```

## 2. Classify Actions

### Python API

```python
from flashvideo import ActionClassifier

clf = ActionClassifier()
results = clf.classify("video.mp4", top_k=5)
for label, score in results:
    print(f"{score:.4f}  {label}")
```

### CLI

```bash
flashvideo classify --video input.mp4 --top-k 5
```

## 3. Simulate a Scene

### Python API

```python
from flashvideo import SceneSimulator

sim = SceneSimulator()
sim.simulate("a robot arm picking up a block", output="sim.mp4", num_frames=32)
```

### CLI

```bash
flashvideo simulate --prompt "a robot arm" --output sim.mp4 --frames 32
```

## 4. Benchmark Performance

```bash
flashvideo benchmark --device cuda --frames 16 --resolution 256
```

## 5. Train a Model

```bash
flashvideo train --config configs/flashvideo_generation.yaml
```

## Next Steps

- [Video Generation](Video-Generation.md) — Full generation pipeline docs
- [Action Recognition](Action-Recognition.md) — TimeSformer details
- [World Models](World-Models.md) — Environment simulation
