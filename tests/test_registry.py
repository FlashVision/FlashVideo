"""Tests for the Registry system."""

import pytest

from flashvideo.registry import Registry, MODELS, SCHEDULERS, DATASETS


class TestRegistry:
    def test_register_and_get(self):
        reg = Registry("test")

        @reg.register("Foo")
        class Foo:
            pass

        assert reg.get("Foo") is Foo
        assert "Foo" in reg
        assert len(reg) == 1

    def test_duplicate_raises(self):
        reg = Registry("test")

        @reg.register("Bar")
        class Bar:
            pass

        with pytest.raises(KeyError, match="already registered"):

            @reg.register("Bar")
            class Bar2:
                pass

    def test_missing_key_raises(self):
        reg = Registry("test")
        with pytest.raises(KeyError, match="not found"):
            reg.get("NonExistent")

    def test_build(self):
        reg = Registry("test")

        @reg.register("Adder")
        class Adder:
            def __init__(self, a, b):
                self.result = a + b

        obj = reg.build("Adder", a=1, b=2)
        assert obj.result == 3

    def test_register_module(self):
        reg = Registry("test")

        class MyMod:
            pass

        reg.register_module("MyMod", MyMod)
        assert reg.get("MyMod") is MyMod

    def test_global_registries_populated(self):
        assert "VideoDiT" in MODELS
        assert "VideoViT" in MODELS
        assert "TimeSformer" in MODELS
        assert "WorldModel" in MODELS

        assert "DDPM" in SCHEDULERS
        assert "DDIM" in SCHEDULERS
        assert "DPM++" in SCHEDULERS

        assert "folder" in DATASETS
        assert "kinetics" in DATASETS

    def test_repr(self):
        reg = Registry("demo")

        @reg.register("A")
        class A:
            pass

        assert "demo" in repr(reg)
        assert "A" in repr(reg)
