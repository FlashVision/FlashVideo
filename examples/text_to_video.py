"""Example: Text-to-video generation with FlashVideo.

Generates a short video clip from a text prompt using the Video DiT model.

Usage:
    python examples/text_to_video.py --prompt "a sunset over the ocean" --steps 20
"""

from __future__ import annotations

import argparse

import torch

from flashvideo.solutions.video_generator import VideoGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashVideo — Text-to-Video Generation")
    parser.add_argument("--prompt", type=str, default="a cat playing in a garden")
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print(f"Generating video: '{args.prompt}'")
    print(f"  Frames: {args.frames}, Steps: {args.steps}, Seed: {args.seed}")

    gen = VideoGenerator(device=args.device)
    path = gen.generate(
        prompt=args.prompt,
        output=args.output,
        num_frames=args.frames,
        num_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    print(f"Done → {path}")


if __name__ == "__main__":
    main()
