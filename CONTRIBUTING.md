# Contributing to FlashVideo

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/FlashVision/FlashVideo.git
cd FlashVideo
pip install -e ".[dev,all]"
```

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes
3. Run lint: `ruff check flashvideo/`
4. Run tests: `pytest tests/ -v`
5. Commit and push
6. Open a Pull Request

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting (line length: 120)
- Type hints are encouraged
- Docstrings for all public functions (Google style)
- No hardcoded file paths — use relative or configurable paths

## Adding a New Architecture

1. Create `flashvideo/models/architectures/your_model.py`
2. Inherit from `nn.Module` and implement `forward()`
3. Register with `@MODELS.register("YourModel")`
4. Add to `flashvideo/models/architectures/__init__.py`

## Adding a New Scheduler

1. Create a scheduler class in `flashvideo/generation/schedulers.py`
2. Inherit from `BaseScheduler` and implement `set_timesteps()`, `step()`, `add_noise()`
3. Register with `@SCHEDULERS.register("YourScheduler")`

## Commit Messages

Use clear, descriptive messages:
- `Add TimeSformer temporal attention`
- `Fix scheduler timestep alignment for video`
- `Update README with world model examples`

## Reporting Issues

- Use GitHub Issues
- Include: Python version, PyTorch version, GPU info, error traceback
- Run `flashvideo settings` and paste the output

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
