"""Example: Video understanding with FlashVideo.

Demonstrates video classification and temporal event detection using
Video ViT and temporal modeling modules.

Usage:
    python examples/video_understanding.py --video input.mp4
"""

from __future__ import annotations

import argparse

import torch

from flashvideo.models.architectures.video_vit import VideoViT
from flashvideo.understanding.classification import VideoClassifier
from flashvideo.understanding.temporal import EventDetector, TemporalModeling


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashVideo — Video Understanding")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print("=== Video Understanding Demo ===\n")

    # Build a small VideoViT
    model = VideoViT(
        in_channels=3,
        num_classes=10,
        embed_dim=128,
        depth=4,
        num_heads=4,
        tubelet_size=(2, 16, 16),
        num_frames=8,
        image_size=128,
    )

    # Classification
    print("1. Video Classification")
    dummy_video = torch.randn(1, 3, 8, 128, 128)
    output = model(dummy_video)
    probs = torch.softmax(output["logits"], dim=-1)
    top_val, top_idx = probs.topk(3)
    for v, i in zip(top_val.squeeze().tolist(), top_idx.squeeze().tolist()):
        print(f"   Class {i}: {v:.4f}")

    # Temporal modeling
    print("\n2. Temporal Event Detection")
    features = torch.randn(1, 32, 128)
    detector = EventDetector(dim=128, num_classes=1)
    events = detector.detect(features, threshold=0.5)
    print(f"   Detected {len(events[0])} events")
    for start, end, score in events[0][:5]:
        print(f"   Frame {start}-{end}: score={score:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
