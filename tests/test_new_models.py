"""Tests for new FlashVideo model architectures and components."""

import torch


class TestCogVideoX:
    def test_forward_shape(self):
        from flashvideo.models.architectures.cogvideox import CogVideoX

        model = CogVideoX(
            in_channels=4,
            hidden_size=64,
            depth=2,
            num_heads=4,
            num_experts=2,
            patch_size=(1, 2, 2),
            num_frames=4,
            image_size=8,
            context_dim=32,
        )
        x = torch.randn(2, 4, 4, 8, 8)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)
        assert out.shape == (2, 4, 4, 8, 8)

    def test_with_text_context(self):
        from flashvideo.models.architectures.cogvideox import CogVideoX

        model = CogVideoX(
            in_channels=4,
            hidden_size=64,
            depth=2,
            num_heads=4,
            num_experts=2,
            patch_size=(1, 2, 2),
            num_frames=4,
            image_size=8,
            context_dim=32,
        )
        x = torch.randn(1, 4, 4, 8, 8)
        t = torch.randint(0, 1000, (1,))
        ctx = torch.randn(1, 10, 32)
        out = model(x, t, context=ctx)
        assert out.shape == (1, 4, 4, 8, 8)

    def test_causal_vae(self):
        from flashvideo.models.architectures.cogvideox import CausalVAEEncoder, CausalVAEDecoder

        enc = CausalVAEEncoder(in_channels=3, latent_dim=8, base_channels=32)
        dec = CausalVAEDecoder(out_channels=3, latent_dim=8, base_channels=32)

        x = torch.randn(1, 3, 4, 32, 32)
        mu, logvar = enc(x)
        assert mu.shape[1] == 8
        recon = dec(mu)
        assert recon.shape[1] == 3


class TestVideoMAE:
    def test_pretrain_forward(self):
        from flashvideo.models.architectures.video_mae import VideoMAE

        model = VideoMAE(
            in_channels=3,
            embed_dim=64,
            encoder_depth=2,
            decoder_dim=32,
            decoder_depth=1,
            num_heads=4,
            decoder_heads=2,
            patch_size=8,
            tubelet_length=2,
            num_frames=4,
            image_size=32,
            mask_ratio=0.75,
        )
        x = torch.randn(2, 3, 4, 32, 32)
        out = model(x, pretrain=True)
        assert "loss" in out
        assert "mask" in out
        assert out["loss"].dim() == 0

    def test_finetune_forward(self):
        from flashvideo.models.architectures.video_mae import VideoMAE

        model = VideoMAE(
            in_channels=3,
            num_classes=10,
            embed_dim=64,
            encoder_depth=2,
            decoder_dim=32,
            decoder_depth=1,
            num_heads=4,
            decoder_heads=2,
            patch_size=8,
            tubelet_length=2,
            num_frames=4,
            image_size=32,
        )
        x = torch.randn(2, 3, 4, 32, 32)
        out = model(x, pretrain=False)
        assert "logits" in out
        assert out["logits"].shape == (2, 10)


class TestSlowFast:
    def test_forward(self):
        from flashvideo.models.architectures.slowfast import SlowFast

        model = SlowFast(
            in_channels=3,
            num_classes=10,
            slow_channels=16,
            fast_channels=4,
            alpha=4,
            layers=(1, 1, 1, 1),
        )
        x = torch.randn(2, 3, 8, 32, 32)
        out = model(x)
        assert "logits" in out
        assert out["logits"].shape == (2, 10)
        assert "features" in out


class TestVideoSuperResolution:
    def test_forward(self):
        from flashvideo.generation.video_sr import VideoSuperResolution

        model = VideoSuperResolution(in_channels=3, mid_channels=16, num_blocks=2, scale_factor=2)
        lr = torch.randn(1, 4, 3, 16, 16)
        sr = model(lr)
        assert sr.shape == (1, 4, 3, 32, 32)


class TestFrameInterpolation:
    def test_single_pair(self):
        from flashvideo.generation.frame_interpolation import FrameInterpolator

        model = FrameInterpolator(in_channels=3, num_scales=2, mid_channels=16)
        img0 = torch.randn(1, 3, 32, 32)
        img1 = torch.randn(1, 3, 32, 32)
        out = model(img0, img1, timestep=0.5)
        assert "output" in out
        assert out["output"].shape == (1, 3, 32, 32)

    def test_interpolate_sequence(self):
        from flashvideo.generation.frame_interpolation import FrameInterpolator

        model = FrameInterpolator(in_channels=3, num_scales=2, mid_channels=16)
        frames = torch.randn(4, 3, 32, 32)
        result = model.interpolate_sequence(frames, multiplier=2)
        assert result.shape[0] == 7  # (4-1)*2 + 1


class TestKineticsDataset:
    def test_registry(self):
        from flashvideo.registry import DATASETS

        assert "kinetics_full" in DATASETS

    def test_instantiation(self, tmp_path):
        from flashvideo.data.kinetics import KineticsFullDataset

        train_dir = tmp_path / "train" / "class_a"
        train_dir.mkdir(parents=True)
        dataset = KineticsFullDataset(root=str(tmp_path), split="train")
        assert dataset.num_classes >= 0


class TestRegistration:
    def test_video_models_registered(self):
        from flashvideo.registry import MODELS

        assert "CogVideoX" in MODELS
        assert "VideoMAE" in MODELS
        assert "SlowFast" in MODELS
