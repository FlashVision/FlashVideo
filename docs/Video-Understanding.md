# Video Understanding

FlashVideo provides modules for comprehensive video understanding:

## Video Classification

Classify videos into action or scene categories:

```python
from flashvideo.models.architectures.video_vit import VideoViT
from flashvideo.understanding.classification import VideoClassifier

model = VideoViT(num_classes=400, embed_dim=768, depth=12)
classifier = VideoClassifier(model=model, num_frames=16)

results = classifier.classify("video.mp4", top_k=5)
```

## Video Captioning

Generate natural-language descriptions:

```python
from flashvideo.understanding.captioning import VideoCaptioner

captioner = VideoCaptioner(vision_encoder=model)
caption = captioner.caption("video.mp4")
```

## Temporal Grounding

Locate moments in a video matching a text query:

```python
from flashvideo.understanding.grounding import TemporalGrounding

grounder = TemporalGrounding(visual_dim=768, text_dim=768)
spans = grounder.ground(visual_features, text_features, fps=30.0)
# Returns: [(start_sec, end_sec, confidence), ...]
```

## Event Detection

Detect action boundaries and events of interest:

```python
from flashvideo.understanding.temporal import EventDetector

detector = EventDetector(dim=768)
events = detector.detect(features, threshold=0.5)
# Returns: [(start_frame, end_frame, score), ...]
```

## Video ViT Architecture

The Video Vision Transformer (ViViT) uses tubelet embeddings to jointly capture spatial and temporal information:

1. **Tubelet Embedding** — 3D convolution extracts `(t, h, w)` tubelets
2. **CLS Token** — Prepended learnable classification token
3. **Positional Embedding** — Learnable 1D positions
4. **Transformer Blocks** — Standard pre-norm self-attention + MLP
5. **Classification Head** — Linear projection from CLS token

## Datasets

| Dataset | Classes | Videos | Resolution |
|---------|---------|--------|------------|
| Kinetics-400 | 400 | ~300K | 256px |
| Kinetics-700 | 700 | ~650K | 256px |
| Something-Something V2 | 174 | ~220K | 240px |
| WebVid-10M | N/A | 10.7M | Various |
