"""Tests for FlashVideo model architectures."""

import pytest
import torch

from flashvideo.models.architectures.video_dit import VideoDiT
from flashvideo.models.architectures.video_vit import VideoViT
from flashvideo.models.architectures.timesformer import TimeSformer
from flashvideo.models.architectures.world_model import WorldModelTransformer
from flashvideo.models.flashvideo_model import FlashVideoModel
from flashvideo.models.lora import LoRALinear, apply_lora


class TestVideoDiT:
    def test_forward_shape(self):
        model = VideoDiT(
            in_channels=4, hidden_size=64, depth=2, num_heads=4,
            patch_size=(1, 2, 2), num_frames=4, image_size=16,
        )
        x = torch.randn(2, 4, 4, 16, 16)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)
        assert out.shape == (2, 4, 4, 16, 16)

    def test_with_context(self):
        model = VideoDiT(
            in_channels=4, hidden_size=64, depth=2, num_heads=4,
            patch_size=(1, 2, 2), num_frames=4, image_size=16,
            context_dim=32,
        )
        x = torch.randn(1, 4, 4, 16, 16)
        t = torch.randint(0, 1000, (1,))
        ctx = torch.randn(1, 10, 32)
        out = model(x, t, context=ctx)
        assert out.shape == (1, 4, 4, 16, 16)


class TestVideoViT:
    def test_classification(self):
        model = VideoViT(
            in_channels=3, num_classes=10, embed_dim=64, depth=2,
            num_heads=4, tubelet_size=(2, 8, 8), num_frames=4, image_size=32,
        )
        x = torch.randn(2, 3, 4, 32, 32)
        out = model(x)
        assert "logits" in out
        assert "features" in out
        assert out["logits"].shape == (2, 10)
        assert out["features"].shape == (2, 64)


class TestTimeSformer:
    def test_forward(self):
        model = TimeSformer(
            in_channels=3, num_classes=10, embed_dim=64, depth=2,
            num_heads=4, patch_size=8, num_frames=4, image_size=32,
        )
        x = torch.randn(2, 3, 4, 32, 32)
        out = model(x)
        assert out["logits"].shape == (2, 10)


class TestWorldModel:
    def test_forward_and_loss(self):
        model = WorldModelTransformer(
            frame_dim=32, hidden_size=64, depth=2, num_heads=4,
            action_dim=8, max_frames=16,
        )
        frames = torch.randn(2, 8, 32)
        actions = torch.randn(2, 8, 8)
        out = model(frames, actions)
        assert "predicted_frames" in out
        assert out["predicted_frames"].shape == (2, 8, 32)
        assert "loss" in out

    def test_rollout(self):
        model = WorldModelTransformer(
            frame_dim=32, hidden_size=64, depth=2, num_heads=4, max_frames=32,
        )
        initial = torch.randn(1, 4, 32)
        result = model.rollout(initial, num_steps=8)
        assert result.shape == (1, 12, 32)


class TestFlashVideoModel:
    def test_build_videovit(self):
        model = FlashVideoModel(
            arch="VideoViT", num_classes=5, embed_dim=64, depth=2,
            num_heads=4, tubelet_size=(2, 8, 8), num_frames=4, image_size=32,
        )
        x = torch.randn(1, 3, 4, 32, 32)
        out = model(x)
        assert out["logits"].shape == (1, 5)
        assert model.num_parameters > 0


class TestLoRA:
    def test_lora_linear(self):
        lora = LoRALinear(32, 64, rank=4, alpha=8.0)
        x = torch.randn(2, 32)
        out = lora(x)
        assert out.shape == (2, 64)

    def test_apply_lora(self):
        model = VideoViT(
            in_channels=3, num_classes=5, embed_dim=64, depth=2,
            num_heads=4, tubelet_size=(2, 8, 8), num_frames=4, image_size=32,
        )
        total_before = sum(p.numel() for p in model.parameters())
        model = apply_lora(model, rank=4)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable < total_before

    def test_merge(self):
        lora = LoRALinear(16, 16, rank=4)
        merged = lora.merge()
        x = torch.randn(1, 16)
        out_lora = lora(x)
        out_merged = merged(x)
        assert torch.allclose(out_lora, out_merged, atol=1e-5)
