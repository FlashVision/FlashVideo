"""FlashVideo — Video Understanding & Generation Framework.

Provides modular components for text-to-video generation, video understanding,
action recognition, and world model simulation.
"""

__version__ = "1.0.0"

from flashvideo.analytics.benchmark import Benchmark
from flashvideo.engine.exporter import Exporter
from flashvideo.engine.predictor import Predictor
from flashvideo.engine.trainer import Trainer
from flashvideo.engine.validator import Validator
from flashvideo.models.flashvideo_model import FlashVideoModel as FlashVideo
from flashvideo.solutions.action_classifier import ActionClassifier
from flashvideo.solutions.scene_simulator import SceneSimulator
from flashvideo.solutions.video_generator import VideoGenerator

__all__ = [
    "__version__",
    "FlashVideo",
    "Trainer",
    "Validator",
    "Predictor",
    "Exporter",
    "VideoGenerator",
    "ActionClassifier",
    "SceneSimulator",
    "Benchmark",
]
