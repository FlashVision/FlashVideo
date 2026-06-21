"""Tests for video generation components."""

import torch

from flashvideo.generation.schedulers import DDPMScheduler, DDIMScheduler, DPMPPScheduler


class TestSchedulers:
    def test_ddpm_add_noise(self):
        s = DDPMScheduler()
        x = torch.randn(2, 4, 4, 8, 8)
        noise = torch.randn_like(x)
        t = torch.tensor([100, 500])
        noisy = s.add_noise(x, noise, t)
        assert noisy.shape == x.shape

    def test_ddim_step(self):
        s = DDIMScheduler()
        s.set_timesteps(10)
        sample = torch.randn(1, 4, 4, 8, 8)
        noise_pred = torch.randn_like(sample)
        result = s.step(noise_pred, 500, sample, eta=0.0)
        assert result.shape == sample.shape

    def test_dpmpp_step(self):
        s = DPMPPScheduler()
        s.set_timesteps(10)
        sample = torch.randn(1, 4, 4, 8, 8)
        noise_pred = torch.randn_like(sample)
        result = s.step(noise_pred, 500, sample)
        assert result.shape == sample.shape

    def test_set_timesteps(self):
        for Cls in [DDPMScheduler, DDIMScheduler, DPMPPScheduler]:
            s = Cls()
            s.set_timesteps(20)
            assert s.timesteps is not None
            assert len(s.timesteps) == 20


class TestMetrics:
    def test_temporal_consistency(self):
        from flashvideo.analytics.metrics import compute_temporal_consistency

        video = torch.randn(16, 3, 32, 32)
        tc = compute_temporal_consistency(video)
        assert isinstance(tc, float)
        assert tc >= 0

    def test_inception_score(self):
        from flashvideo.analytics.metrics import compute_inception_score

        logits = torch.randn(100, 10)
        mean_is, std_is = compute_inception_score(logits, num_splits=5)
        assert mean_is > 0

    def test_fvd(self):
        from flashvideo.analytics.metrics import compute_fvd

        real = torch.randn(50, 128)
        fake = torch.randn(50, 128)
        fvd = compute_fvd(real, fake)
        assert isinstance(fvd, float)
