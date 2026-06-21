# FAQ

## General

**Q: What GPU do I need?**
A: For inference with small models, a GPU with 4GB+ VRAM is sufficient. For training, we recommend 24GB+ (RTX 3090, A5000, A100). CPU inference is supported but slow.

**Q: Does FlashVideo work on CPU?**
A: Yes. All models run on CPU for development and testing. Set `--device cpu` in the CLI.

**Q: What video formats are supported?**
A: MP4, AVI, MOV, MKV, and WebM via OpenCV and decord.

## Video Generation

**Q: How long are generated videos?**
A: Default is 16 frames at 8 fps (2 seconds). You can increase `--frames` for longer videos, limited by GPU memory.

**Q: Can I generate HD video?**
A: The default resolution is 256x256. Higher resolutions (512, 1024) require more VRAM and longer inference time.

**Q: How does classifier-free guidance work?**
A: CFG amplifies the text-conditioned signal. Higher `--guidance-scale` (7-15) produces more prompt-adherent videos at the cost of diversity. Set to 1.0 for unconditional generation.

## Action Recognition

**Q: What datasets are supported?**
A: Kinetics-400/600/700 and Something-Something V2 out of the box. Any folder of videos with class subdirectories works via `FolderVideoDataset`.

**Q: How many frames should I use?**
A: 8 frames for TimeSformer, 16 frames for VideoViT is the default. More frames capture longer temporal context but increase compute.

## World Models

**Q: What is the physics prior?**
A: A soft regularisation that penalises large energy changes between consecutive states, encouraging smoother and more physically-plausible predictions.

**Q: Can I use custom actions?**
A: Yes. Actions can be continuous vectors or discrete IDs. Set `action_dim` in the world model config.

## Training

**Q: How do I use mixed precision?**
A: Set `mixed_precision: true` in your YAML config. It's enabled by default.

**Q: How do I use LoRA?**
A: Call `apply_lora(model, rank=8)` to freeze base weights and add low-rank adapters. Only ~1-2% of parameters will be trainable.

## Troubleshooting

**Q: `ModuleNotFoundError: No module named 'decord'`**
A: Install decord: `pip install decord`. If it fails, the system will fall back to OpenCV for video reading.

**Q: CUDA out of memory**
A: Reduce batch size, enable gradient checkpointing, use mixed precision, or reduce model size / resolution.
