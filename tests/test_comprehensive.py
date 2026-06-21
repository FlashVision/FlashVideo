"""Comprehensive tests for FlashVideo covering all architectures, generation, and pipelines."""

import torch
import pytest


# ---------------------------------------------------------------------------
# 1. CogVideoX
# ---------------------------------------------------------------------------
class TestCogVideoXComprehensive:
    def test_forward_shape(self):
        from flashvideo.models.architectures.cogvideox import CogVideoX

        m = CogVideoX(
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
        out = m(x, t)
        assert out.shape == (1, 4, 4, 8, 8)

    def test_with_text_context(self):
        from flashvideo.models.architectures.cogvideox import CogVideoX

        m = CogVideoX(
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
        out = m(torch.randn(1, 4, 4, 8, 8), torch.randint(0, 1000, (1,)), context=torch.randn(1, 10, 32))
        assert out.shape == (1, 4, 4, 8, 8)

    def test_causal_vae(self):
        from flashvideo.models.architectures.cogvideox import CausalVAEEncoder, CausalVAEDecoder

        enc = CausalVAEEncoder(in_channels=3, latent_dim=8, base_channels=32)
        dec = CausalVAEDecoder(out_channels=3, latent_dim=8, base_channels=32)
        x = torch.randn(1, 3, 4, 16, 16)
        mu, logvar = enc(x)
        assert mu.shape[1] == 8
        recon = dec(mu)
        assert recon.shape[1] == 3

    def test_registered(self):
        from flashvideo.registry import MODELS

        assert "CogVideoX" in MODELS


# ---------------------------------------------------------------------------
# 2. VideoMAE
# ---------------------------------------------------------------------------
class TestVideoMAEComprehensive:
    def test_pretrain(self):
        from flashvideo.models.architectures.video_mae import VideoMAE

        m = VideoMAE(
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
        out = m(torch.randn(2, 3, 4, 32, 32), pretrain=True)
        assert "loss" in out and "mask" in out
        assert out["loss"].dim() == 0

    def test_finetune(self):
        from flashvideo.models.architectures.video_mae import VideoMAE

        m = VideoMAE(
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
        out = m(torch.randn(2, 3, 4, 32, 32), pretrain=False)
        assert out["logits"].shape == (2, 10)

    def test_registered(self):
        from flashvideo.registry import MODELS

        assert "VideoMAE" in MODELS


# ---------------------------------------------------------------------------
# 3. SlowFast
# ---------------------------------------------------------------------------
class TestSlowFastComprehensive:
    def test_forward(self):
        from flashvideo.models.architectures.slowfast import SlowFast

        m = SlowFast(in_channels=3, num_classes=10, slow_channels=16, fast_channels=4, alpha=4, layers=(1, 1, 1, 1))
        out = m(torch.randn(2, 3, 8, 32, 32))
        assert out["logits"].shape == (2, 10)
        assert "features" in out

    def test_registered(self):
        from flashvideo.registry import MODELS

        assert "SlowFast" in MODELS


# ---------------------------------------------------------------------------
# 4. Video DiT
# ---------------------------------------------------------------------------
class TestVideoDiTComprehensive:
    def test_forward(self):
        from flashvideo.models.architectures.video_dit import VideoDiT

        m = VideoDiT(
            in_channels=4, hidden_size=64, depth=2, num_heads=4, patch_size=(1, 2, 2), num_frames=4, image_size=16
        )
        out = m(torch.randn(1, 4, 4, 16, 16), torch.randint(0, 1000, (1,)))
        assert out.shape == (1, 4, 4, 16, 16)

    def test_with_context(self):
        from flashvideo.models.architectures.video_dit import VideoDiT

        m = VideoDiT(
            in_channels=4,
            hidden_size=64,
            depth=2,
            num_heads=4,
            patch_size=(1, 2, 2),
            num_frames=4,
            image_size=16,
            context_dim=32,
        )
        out = m(torch.randn(1, 4, 4, 16, 16), torch.randint(0, 1000, (1,)), context=torch.randn(1, 10, 32))
        assert out.shape == (1, 4, 4, 16, 16)

    def test_batch(self):
        from flashvideo.models.architectures.video_dit import VideoDiT

        m = VideoDiT(
            in_channels=4, hidden_size=64, depth=2, num_heads=4, patch_size=(1, 2, 2), num_frames=4, image_size=16
        )
        out = m(torch.randn(2, 4, 4, 16, 16), torch.randint(0, 1000, (2,)))
        assert out.shape[0] == 2


# ---------------------------------------------------------------------------
# 5. VideoViT
# ---------------------------------------------------------------------------
class TestVideoViTComprehensive:
    def test_classification(self):
        from flashvideo.models.architectures.video_vit import VideoViT

        m = VideoViT(
            in_channels=3,
            num_classes=10,
            embed_dim=64,
            depth=2,
            num_heads=4,
            tubelet_size=(2, 8, 8),
            num_frames=4,
            image_size=32,
        )
        out = m(torch.randn(2, 3, 4, 32, 32))
        assert out["logits"].shape == (2, 10)
        assert out["features"].shape == (2, 64)


# ---------------------------------------------------------------------------
# 6. TimeSformer
# ---------------------------------------------------------------------------
class TestTimeSformerComprehensive:
    def test_forward(self):
        from flashvideo.models.architectures.timesformer import TimeSformer

        m = TimeSformer(
            in_channels=3, num_classes=10, embed_dim=64, depth=2, num_heads=4, patch_size=8, num_frames=4, image_size=32
        )
        out = m(torch.randn(2, 3, 4, 32, 32))
        assert out["logits"].shape == (2, 10)


# ---------------------------------------------------------------------------
# 7. World Model
# ---------------------------------------------------------------------------
class TestWorldModelComprehensive:
    def test_forward_loss(self):
        from flashvideo.models.architectures.world_model import WorldModelTransformer

        m = WorldModelTransformer(frame_dim=32, hidden_size=64, depth=2, num_heads=4, action_dim=8, max_frames=16)
        out = m(torch.randn(2, 8, 32), torch.randn(2, 8, 8))
        assert out["predicted_frames"].shape == (2, 8, 32)
        assert "loss" in out

    def test_rollout(self):
        from flashvideo.models.architectures.world_model import WorldModelTransformer

        m = WorldModelTransformer(frame_dim=32, hidden_size=64, depth=2, num_heads=4, max_frames=32)
        result = m.rollout(torch.randn(1, 4, 32), num_steps=8)
        assert result.shape == (1, 12, 32)

    def test_no_action(self):
        from flashvideo.models.architectures.world_model import WorldModelTransformer

        m = WorldModelTransformer(frame_dim=32, hidden_size=64, depth=2, num_heads=4, max_frames=16)
        out = m(torch.randn(1, 8, 32))
        assert "predicted_frames" in out


# ---------------------------------------------------------------------------
# 8. Video Generation: schedulers
# ---------------------------------------------------------------------------
class TestSchedulersComprehensive:
    def test_ddpm_add_noise(self):
        from flashvideo.generation.schedulers import DDPMScheduler

        s = DDPMScheduler()
        x = torch.randn(1, 4, 4, 8, 8)
        noisy = s.add_noise(x, torch.randn_like(x), torch.tensor([100]))
        assert noisy.shape == x.shape

    def test_ddim_step(self):
        from flashvideo.generation.schedulers import DDIMScheduler

        s = DDIMScheduler()
        s.set_timesteps(10)
        sample = torch.randn(1, 4, 4, 8, 8)
        result = s.step(torch.randn_like(sample), 500, sample, eta=0.0)
        assert result.shape == sample.shape

    def test_dpmpp_step(self):
        from flashvideo.generation.schedulers import DPMPPScheduler

        s = DPMPPScheduler()
        s.set_timesteps(10)
        sample = torch.randn(1, 4, 4, 8, 8)
        result = s.step(torch.randn_like(sample), 500, sample)
        assert result.shape == sample.shape

    def test_set_timesteps(self):
        from flashvideo.generation.schedulers import DDPMScheduler, DDIMScheduler, DPMPPScheduler

        for Cls in [DDPMScheduler, DDIMScheduler, DPMPPScheduler]:
            s = Cls()
            s.set_timesteps(20)
            assert len(s.timesteps) == 20


# ---------------------------------------------------------------------------
# 9. Text-to-Video pipeline
# ---------------------------------------------------------------------------
class TestTextToVideoPipelineComprehensive:
    def test_pipeline_basic(self):
        from flashvideo.generation.text_to_video import TextToVideoPipeline
        from flashvideo.models.architectures.video_dit import VideoDiT

        model = VideoDiT(
            in_channels=4,
            hidden_size=64,
            depth=2,
            num_heads=4,
            patch_size=(1, 2, 2),
            num_frames=4,
            image_size=8,
            context_dim=32,
        )
        pipe = TextToVideoPipeline(model=model, device="cpu")
        video = pipe(
            prompt="test",
            num_frames=4,
            height=16,
            width=16,
            num_steps=2,
            guidance_scale=1.0,
            context=torch.randn(1, 10, 32),
        )
        assert video.ndim == 4
        assert video.dtype == torch.uint8

    def test_latents_to_frames(self):
        from flashvideo.generation.text_to_video import TextToVideoPipeline

        latents = torch.randn(1, 3, 4, 8, 8)
        frames = TextToVideoPipeline._latents_to_frames(latents)
        assert frames.dtype == torch.uint8
        assert frames.shape[-1] == 3


# ---------------------------------------------------------------------------
# 10. Video SR
# ---------------------------------------------------------------------------
class TestVideoSRComprehensive:
    def test_forward(self):
        from flashvideo.generation.video_sr import VideoSuperResolution

        m = VideoSuperResolution(in_channels=3, mid_channels=16, num_blocks=2, scale_factor=2)
        lr = torch.randn(1, 4, 3, 16, 16)
        sr = m(lr)
        assert sr.shape == (1, 4, 3, 32, 32)

    def test_scale_4x(self):
        from flashvideo.generation.video_sr import VideoSuperResolution

        m = VideoSuperResolution(in_channels=3, mid_channels=16, num_blocks=2, scale_factor=4)
        lr = torch.randn(1, 2, 3, 8, 8)
        sr = m(lr)
        assert sr.shape == (1, 2, 3, 32, 32)


# ---------------------------------------------------------------------------
# 11. Frame Interpolation
# ---------------------------------------------------------------------------
class TestFrameInterpolationComprehensive:
    def test_single_pair(self):
        from flashvideo.generation.frame_interpolation import FrameInterpolator

        m = FrameInterpolator(in_channels=3, num_scales=2, mid_channels=16)
        out = m(torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32), timestep=0.5)
        assert out["output"].shape == (1, 3, 32, 32)

    def test_sequence(self):
        from flashvideo.generation.frame_interpolation import FrameInterpolator

        m = FrameInterpolator(in_channels=3, num_scales=2, mid_channels=16)
        frames = torch.randn(4, 3, 32, 32)
        result = m.interpolate_sequence(frames, multiplier=2)
        assert result.shape[0] == 7


# ---------------------------------------------------------------------------
# 12. Video Understanding: classification, temporal
# ---------------------------------------------------------------------------
class TestVideoClassifierComprehensive:
    def test_classify_tensor(self):
        from flashvideo.models.architectures.video_vit import VideoViT
        from flashvideo.understanding.classification import VideoClassifier

        backbone = VideoViT(
            in_channels=3,
            num_classes=5,
            embed_dim=64,
            depth=2,
            num_heads=4,
            tubelet_size=(2, 8, 8),
            num_frames=4,
            image_size=32,
        )
        clf = VideoClassifier(
            model=backbone, class_names=["a", "b", "c", "d", "e"], num_frames=4, image_size=32, device="cpu"
        )
        video = torch.randn(4, 3, 32, 32)
        results = clf.classify(video, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)

    def test_predict_batch(self):
        from flashvideo.models.architectures.video_vit import VideoViT
        from flashvideo.understanding.classification import VideoClassifier

        backbone = VideoViT(
            in_channels=3,
            num_classes=5,
            embed_dim=64,
            depth=2,
            num_heads=4,
            tubelet_size=(2, 8, 8),
            num_frames=4,
            image_size=32,
        )
        clf = VideoClassifier(model=backbone, num_frames=4, image_size=32, device="cpu")
        videos = [torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)]
        results = clf.predict_batch(videos, top_k=2)
        assert len(results) == 2


class TestTemporalModelingComprehensive:
    def test_temporal_modeling(self):
        from flashvideo.understanding.temporal import TemporalModeling

        tm = TemporalModeling(dim=32, scales=(3, 5))
        x = torch.randn(2, 16, 32)
        out = tm(x)
        assert out.shape == (2, 16, 32)

    def test_event_detector(self):
        from flashvideo.understanding.temporal import EventDetector

        ed = EventDetector(dim=32, num_classes=1)
        feat = torch.randn(2, 16, 32)
        out = ed(feat)
        assert out["event_scores"].shape == (2, 16, 1)
        assert out["boundary_scores"].shape == (2, 16, 2)

    def test_event_detect_segments(self):
        from flashvideo.understanding.temporal import EventDetector

        ed = EventDetector(dim=32, num_classes=1)
        feat = torch.randn(1, 16, 32)
        segments = ed.detect(feat, threshold=0.5)
        assert isinstance(segments, list) and isinstance(segments[0], list)


# ---------------------------------------------------------------------------
# 13. World Model Dynamics
# ---------------------------------------------------------------------------
class TestDynamicsComprehensive:
    def test_dynamics_model(self):
        from flashvideo.world_models.dynamics import DynamicsModel

        m = DynamicsModel(state_dim=32, action_dim=8, hidden_dim=64)
        out = m(torch.randn(2, 32), torch.randn(2, 8))
        assert out["next_state"].shape == (2, 32)

    def test_dynamics_with_target(self):
        from flashvideo.world_models.dynamics import DynamicsModel

        m = DynamicsModel(state_dim=32, action_dim=8, hidden_dim=64)
        out = m(torch.randn(2, 32), torch.randn(2, 8), target=torch.randn(2, 32))
        assert "loss" in out

    def test_dynamics_rollout(self):
        from flashvideo.world_models.dynamics import DynamicsModel

        m = DynamicsModel(state_dim=32, action_dim=8, hidden_dim=64)
        traj = m.rollout(torch.randn(1, 32), torch.randn(1, 5, 8))
        assert traj.shape == (1, 6, 32)

    def test_physics_prior(self):
        from flashvideo.world_models.dynamics import PhysicsPrior

        pp = PhysicsPrior(state_dim=32)
        e = pp.compute_energy(torch.randn(5, 32))
        assert e.shape == (5, 1)
        loss = pp.conservation_loss(torch.randn(1, 10, 32))
        assert loss.dim() == 0

    def test_physics_loss(self):
        from flashvideo.world_models.dynamics import DynamicsModel

        m = DynamicsModel(state_dim=32, action_dim=8, hidden_dim=64, use_physics_prior=True)
        traj = torch.randn(1, 10, 32)
        loss = m.physics_loss(traj)
        assert loss.dim() == 0

    def test_no_physics(self):
        from flashvideo.world_models.dynamics import DynamicsModel

        m = DynamicsModel(state_dim=32, action_dim=8, hidden_dim=64, use_physics_prior=False)
        loss = m.physics_loss(torch.randn(1, 5, 32))
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# 14. Kinetics Dataset
# ---------------------------------------------------------------------------
class TestKineticsComprehensive:
    def test_registered(self):
        from flashvideo.registry import DATASETS

        assert "kinetics_full" in DATASETS

    def test_instantiation(self, tmp_path):
        from flashvideo.data.kinetics import KineticsFullDataset

        (tmp_path / "train" / "class_a").mkdir(parents=True)
        ds = KineticsFullDataset(root=str(tmp_path), split="train")
        assert ds.num_classes >= 0

    def test_frame_cache(self, tmp_path):
        from flashvideo.data.kinetics import FrameCache

        cache = FrameCache(str(tmp_path / "cache"))
        assert cache.get("video.mp4", [0, 1, 2]) is None
        cache.put("video.mp4", [0, 1, 2], torch.randn(3, 3, 32, 32))
        result = cache.get("video.mp4", [0, 1, 2])
        assert result is not None

    def test_frame_cache_disabled(self, tmp_path):
        from flashvideo.data.kinetics import FrameCache

        cache = FrameCache(str(tmp_path / "cache"), enabled=False)
        cache.put("video.mp4", [0], torch.randn(1, 3, 32, 32))
        assert cache.get("video.mp4", [0]) is None

    def test_frame_cache_clear(self, tmp_path):
        from flashvideo.data.kinetics import FrameCache

        cache = FrameCache(str(tmp_path / "cache"))
        cache.put("video.mp4", [0], torch.randn(1, 3, 32, 32))
        cache.clear()
        assert cache.get("video.mp4", [0]) is None

    def test_class_weights(self, tmp_path):
        from flashvideo.data.kinetics import KineticsFullDataset

        for cls in ["a", "b"]:
            d = tmp_path / "train" / cls
            d.mkdir(parents=True)
        ds = KineticsFullDataset(root=str(tmp_path), split="train")
        if ds.num_classes > 0:
            w = ds.get_class_weights()
            assert w.shape[0] == ds.num_classes


# ---------------------------------------------------------------------------
# 15. FlashVideo model wrapper
# ---------------------------------------------------------------------------
class TestFlashVideoModelComprehensive:
    def test_build_videovit(self):
        from flashvideo.models.flashvideo_model import FlashVideoModel

        m = FlashVideoModel(
            arch="VideoViT",
            num_classes=5,
            embed_dim=64,
            depth=2,
            num_heads=4,
            tubelet_size=(2, 8, 8),
            num_frames=4,
            image_size=32,
        )
        out = m(torch.randn(1, 3, 4, 32, 32))
        assert out["logits"].shape == (1, 5)
        assert m.num_parameters > 0


# ---------------------------------------------------------------------------
# 16. LoRA
# ---------------------------------------------------------------------------
class TestLoRAComprehensive:
    def test_lora_linear(self):
        from flashvideo.models.lora import LoRALinear

        lora = LoRALinear(32, 64, rank=4, alpha=8.0)
        out = lora(torch.randn(2, 32))
        assert out.shape == (2, 64)

    def test_apply_lora(self):
        from flashvideo.models.architectures.video_vit import VideoViT
        from flashvideo.models.lora import apply_lora

        m = VideoViT(
            in_channels=3,
            num_classes=5,
            embed_dim=64,
            depth=2,
            num_heads=4,
            tubelet_size=(2, 8, 8),
            num_frames=4,
            image_size=32,
        )
        total_before = sum(p.numel() for p in m.parameters())
        m = apply_lora(m, rank=4)
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert trainable < total_before

    def test_merge(self):
        from flashvideo.models.lora import LoRALinear

        lora = LoRALinear(16, 16, rank=4)
        merged = lora.merge()
        x = torch.randn(1, 16)
        assert torch.allclose(lora(x), merged(x), atol=1e-5)


# ---------------------------------------------------------------------------
# 17. Metrics
# ---------------------------------------------------------------------------
class TestMetricsComprehensive:
    def test_temporal_consistency(self):
        from flashvideo.analytics.metrics import compute_temporal_consistency

        tc = compute_temporal_consistency(torch.randn(8, 3, 16, 16))
        assert isinstance(tc, float) and tc >= 0

    def test_inception_score(self):
        from flashvideo.analytics.metrics import compute_inception_score

        mean_is, std_is = compute_inception_score(torch.randn(50, 10), num_splits=5)
        assert mean_is > 0

    def test_fvd(self):
        from flashvideo.analytics.metrics import compute_fvd

        fvd = compute_fvd(torch.randn(50, 64), torch.randn(50, 64))
        assert isinstance(fvd, float)


# ---------------------------------------------------------------------------
# 18. CLI
# ---------------------------------------------------------------------------
class TestCLIComprehensive:
    def test_build_parser(self):
        from flashvideo.cli import build_parser

        parser = build_parser()
        assert parser is not None

    def test_version_subcommand(self):
        from flashvideo.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["version"])
        assert args.command == "version"

    def test_generate_subcommand(self):
        from flashvideo.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["generate", "--prompt", "test"])
        assert args.command == "generate" and args.prompt == "test"


# ---------------------------------------------------------------------------
# 19. Registry
# ---------------------------------------------------------------------------
class TestRegistryComprehensive:
    def test_register_and_get(self):
        from flashvideo.registry import Registry

        reg = Registry("test_comp_v")

        @reg.register("ItemA")
        class A:
            pass

        assert reg.get("ItemA") is A

    def test_duplicate_raises(self):
        from flashvideo.registry import Registry

        reg = Registry("test_dup_v")

        @reg.register("X")
        class X:
            pass

        with pytest.raises(KeyError):

            @reg.register("X")
            class Y:
                pass

    def test_missing_raises(self):
        from flashvideo.registry import Registry

        reg = Registry("test_miss_v")
        with pytest.raises(KeyError):
            reg.get("nope")

    def test_build(self):
        from flashvideo.registry import Registry

        reg = Registry("test_build_v")

        @reg.register("Adder")
        class Adder:
            def __init__(self, a, b):
                self.result = a + b

        obj = reg.build("Adder", a=1, b=2)
        assert obj.result == 3

    def test_global_registries(self):
        from flashvideo.registry import MODELS, SCHEDULERS

        assert "VideoDiT" in MODELS
        assert "VideoViT" in MODELS
        assert "DDPM" in SCHEDULERS
        assert "DDIM" in SCHEDULERS


# ---------------------------------------------------------------------------
# 20. Integration: load video → classify/generate → output
# ---------------------------------------------------------------------------
class TestIntegrationComprehensive:
    def test_classify_pipeline(self):
        from flashvideo.models.architectures.video_vit import VideoViT
        from flashvideo.understanding.classification import VideoClassifier

        backbone = VideoViT(
            in_channels=3,
            num_classes=5,
            embed_dim=64,
            depth=2,
            num_heads=4,
            tubelet_size=(2, 8, 8),
            num_frames=4,
            image_size=32,
        )
        clf = VideoClassifier(
            model=backbone,
            class_names=["walk", "run", "jump", "sit", "stand"],
            num_frames=4,
            image_size=32,
            device="cpu",
        )
        video = torch.randn(4, 3, 32, 32)
        results = clf.classify(video, top_k=5)
        assert len(results) == 5
        probs = [r[1] for r in results]
        assert abs(sum(probs) - 1.0) < 0.1

    def test_generation_pipeline(self):
        from flashvideo.generation.text_to_video import TextToVideoPipeline
        from flashvideo.models.architectures.video_dit import VideoDiT

        model = VideoDiT(
            in_channels=4,
            hidden_size=64,
            depth=2,
            num_heads=4,
            patch_size=(1, 2, 2),
            num_frames=4,
            image_size=8,
            context_dim=32,
        )
        pipe = TextToVideoPipeline(model=model, device="cpu")
        video = pipe(
            prompt="test",
            num_frames=4,
            height=16,
            width=16,
            num_steps=2,
            guidance_scale=1.0,
            context=torch.randn(1, 10, 32),
        )
        assert video.dtype == torch.uint8
        assert video.shape[-1] == 3

    def test_world_model_rollout(self):
        from flashvideo.world_models.dynamics import DynamicsModel

        m = DynamicsModel(state_dim=32, action_dim=8)
        initial = torch.randn(1, 32)
        actions = torch.randn(1, 10, 8)
        traj = m.rollout(initial, actions)
        assert traj.shape == (1, 11, 32)

    def test_event_detection_pipeline(self):
        from flashvideo.understanding.temporal import EventDetector

        ed = EventDetector(dim=32, num_classes=1)
        feat = torch.randn(1, 20, 32)
        segments = ed.detect(feat, threshold=0.5)
        assert isinstance(segments, list)
