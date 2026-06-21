"""Performance benchmarking for FlashVideo models."""

from __future__ import annotations

import time
from typing import Optional

import torch

from flashvideo.models.architectures.video_dit import VideoDiT


class Benchmark:
    """Run inference speed and memory benchmarks.

    Usage::

        bench = Benchmark(device="cuda")
        bench.run(num_frames=16, resolution=256, iterations=50)
    """

    def __init__(self, device: str = "auto") -> None:
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")

    def run(
        self,
        model: Optional[torch.nn.Module] = None,
        num_frames: int = 16,
        resolution: int = 256,
        iterations: int = 50,
        batch_size: int = 1,
    ) -> dict:
        """Run benchmark and print results.

        Returns:
            Dictionary of benchmark metrics.
        """
        if model is None:
            model = VideoDiT(
                in_channels=4,
                hidden_size=384,
                depth=6,
                num_heads=6,
                num_frames=num_frames,
                image_size=resolution,
            )

        model = model.to(self.device).eval()
        dummy_video = torch.randn(batch_size, 4, num_frames, resolution // 8, resolution // 8, device=self.device)
        dummy_t = torch.randint(0, 1000, (batch_size,), device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                model(dummy_video, dummy_t)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        times = []
        with torch.no_grad():
            for _ in range(iterations):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(dummy_video, dummy_t)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

        avg_ms = sum(times) / len(times) * 1000
        throughput = batch_size / (sum(times) / len(times))
        peak_mem = 0.0
        if self.device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        params = sum(p.numel() for p in model.parameters())

        results = {
            "device": str(self.device),
            "parameters": params,
            "avg_latency_ms": round(avg_ms, 2),
            "throughput_videos_per_sec": round(throughput, 2),
            "peak_memory_mb": round(peak_mem, 2),
            "num_frames": num_frames,
            "resolution": resolution,
            "iterations": iterations,
        }

        print("=" * 50)
        print("  FlashVideo Benchmark Results")
        print("=" * 50)
        for k, v in results.items():
            print(f"  {k:30s}: {v}")
        print("=" * 50)

        return results
