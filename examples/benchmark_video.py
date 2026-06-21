"""Example: Benchmark FlashVideo model performance.

Measures inference speed, throughput, and memory usage for the
Video DiT model.

Usage:
    python examples/benchmark_video.py --device cuda --iterations 100
"""

from __future__ import annotations

import argparse

from flashvideo.analytics.benchmark import Benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashVideo — Performance Benchmark")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    bench = Benchmark(device=args.device)
    results = bench.run(
        num_frames=args.frames,
        resolution=args.resolution,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
