"""FlashVideo command-line interface."""

from __future__ import annotations

import argparse
import sys


def _version(args: argparse.Namespace) -> None:
    from flashvideo import __version__

    print(f"flashvideo {__version__}")


def _settings(args: argparse.Namespace) -> None:
    import platform

    import numpy
    import torch

    print("FlashVideo Environment")
    print("=" * 40)
    print(f"  Platform:       {platform.platform()}")
    print(f"  Python:         {platform.python_version()}")
    print(f"  PyTorch:        {torch.__version__}")
    cuda = torch.version.cuda if torch.cuda.is_available() else "N/A (CPU)"
    print(f"  CUDA:           {cuda}")
    if torch.cuda.is_available():
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"  GPU Memory:     {mem:.1f} GB")
    print(f"  NumPy:          {numpy.__version__}")
    try:
        import decord

        print(f"  Decord:         {decord.__version__}")
    except Exception:
        print("  Decord:         not installed")
    try:
        import diffusers

        print(f"  Diffusers:      {diffusers.__version__}")
    except Exception:
        print("  Diffusers:      not installed")


def _check(args: argparse.Namespace) -> None:
    checks = []

    def _try_import(name: str) -> bool:
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    for pkg in [
        "torch",
        "torchvision",
        "numpy",
        "cv2",
        "PIL",
        "yaml",
        "tqdm",
        "safetensors",
        "diffusers",
        "transformers",
    ]:
        ok = _try_import(pkg)
        checks.append((pkg, ok))

    try:
        import decord  # noqa: F401

        checks.append(("decord", True))
    except ImportError:
        checks.append(("decord", False))

    all_ok = all(v for _, v in checks)
    for name, ok in checks:
        status = "OK" if ok else "MISSING"
        print(f"  [{status:>7s}] {name}")
    print()
    if all_ok:
        print("All checks passed.")
    else:
        print("Some dependencies are missing. Run: pip install flashvideo[all]")
        sys.exit(1)


def _generate(args: argparse.Namespace) -> None:
    from flashvideo.solutions.video_generator import VideoGenerator

    gen = VideoGenerator(device=args.device)
    gen.generate(
        prompt=args.prompt,
        output=args.output,
        num_frames=args.frames,
        num_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )


def _train(args: argparse.Namespace) -> None:
    from flashvideo.cfg.config import load_config
    from flashvideo.engine.trainer import Trainer

    cfg = load_config(args.config) if args.config else None
    trainer = Trainer(cfg)
    trainer.train()


def _classify(args: argparse.Namespace) -> None:
    from flashvideo.solutions.action_classifier import ActionClassifier

    classifier = ActionClassifier(device=args.device)
    results = classifier.classify(args.video, top_k=args.top_k)
    for label, score in results:
        print(f"  {score:.4f}  {label}")


def _simulate(args: argparse.Namespace) -> None:
    from flashvideo.solutions.scene_simulator import SceneSimulator

    sim = SceneSimulator(device=args.device)
    sim.simulate(
        prompt=args.prompt,
        actions=args.actions.split(",") if args.actions else None,
        output=args.output,
        num_frames=args.frames,
    )


def _export(args: argparse.Namespace) -> None:
    from flashvideo.engine.exporter import Exporter

    exporter = Exporter()
    exporter.export(args.model, args.output, fmt=args.format)


def _benchmark(args: argparse.Namespace) -> None:
    from flashvideo.analytics.benchmark import Benchmark

    bench = Benchmark(device=args.device)
    bench.run(
        num_frames=args.frames,
        resolution=args.resolution,
        iterations=args.iterations,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flashvideo",
        description="FlashVideo — Video Understanding & Generation CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # version
    sub.add_parser("version", help="Show version")

    # settings
    sub.add_parser("settings", help="Show environment info")

    # check
    sub.add_parser("check", help="Verify dependencies")

    # generate
    gen = sub.add_parser("generate", help="Text-to-video generation")
    gen.add_argument("--prompt", type=str, required=True, help="Text prompt")
    gen.add_argument("--output", type=str, default="output.mp4", help="Output file")
    gen.add_argument("--frames", type=int, default=16, help="Number of frames")
    gen.add_argument("--steps", type=int, default=50, help="Diffusion steps")
    gen.add_argument("--guidance-scale", type=float, default=7.5)
    gen.add_argument("--device", type=str, default="auto")

    # train
    tr = sub.add_parser("train", help="Train a video model")
    tr.add_argument("--config", type=str, required=True, help="YAML config path")

    # classify
    cl = sub.add_parser("classify", help="Video action classification")
    cl.add_argument("--video", type=str, required=True, help="Input video file")
    cl.add_argument("--top-k", type=int, default=5, help="Top-K predictions")
    cl.add_argument("--device", type=str, default="auto")

    # simulate
    sim = sub.add_parser("simulate", help="World model scene simulation")
    sim.add_argument("--prompt", type=str, required=True, help="Scene description")
    sim.add_argument("--actions", type=str, default=None, help="Comma-separated actions")
    sim.add_argument("--output", type=str, default="simulation.mp4")
    sim.add_argument("--frames", type=int, default=32)
    sim.add_argument("--device", type=str, default="auto")

    # export
    ex = sub.add_parser("export", help="Export model to ONNX")
    ex.add_argument("--model", type=str, required=True, help="Model checkpoint")
    ex.add_argument("--output", type=str, default="model.onnx")
    ex.add_argument("--format", type=str, default="onnx", choices=["onnx"])

    # benchmark
    bm = sub.add_parser("benchmark", help="Run performance benchmarks")
    bm.add_argument("--device", type=str, default="auto")
    bm.add_argument("--frames", type=int, default=16)
    bm.add_argument("--resolution", type=int, default=256)
    bm.add_argument("--iterations", type=int, default=50)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "version": _version,
        "settings": _settings,
        "check": _check,
        "generate": _generate,
        "train": _train,
        "classify": _classify,
        "simulate": _simulate,
        "export": _export,
        "benchmark": _benchmark,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
