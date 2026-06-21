"""Example: Action recognition with TimeSformer.

Demonstrates divided space-time attention for video action classification.

Usage:
    python examples/action_recognition.py --video input.mp4
"""

from __future__ import annotations

import argparse

import torch

from flashvideo.models.architectures.timesformer import TimeSformer
from flashvideo.understanding.classification import VideoClassifier


KINETICS_SAMPLE_CLASSES = [
    "abseiling",
    "air drumming",
    "answering questions",
    "applauding",
    "applying cream",
    "archery",
    "arm wrestling",
    "arranging flowers",
    "assembling computer",
    "auctioning",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashVideo — Action Recognition")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("=== Action Recognition Demo (TimeSformer) ===\n")

    model = TimeSformer(
        in_channels=3,
        num_classes=len(KINETICS_SAMPLE_CLASSES),
        embed_dim=192,
        depth=4,
        num_heads=4,
        patch_size=16,
        num_frames=8,
        image_size=224,
    )

    classifier = VideoClassifier(
        model=model,
        class_names=KINETICS_SAMPLE_CLASSES,
        num_frames=8,
        image_size=224,
        device=args.device,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    dummy_video = torch.randn(8, 224, 224, 3).to(torch.uint8)
    results = classifier.classify(dummy_video, top_k=args.top_k)

    print("Top predictions:")
    for label, score in results:
        print(f"  {score:.4f}  {label}")

    print("\nDone!")


if __name__ == "__main__":
    main()
