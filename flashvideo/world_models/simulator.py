"""Environment simulator — generate future video frames from a scene description.

Combines a world-model backbone with a latent decoder to render simulated
environments, suitable for robotics, gaming, and autonomous driving.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from flashvideo.models.architectures.world_model import WorldModelTransformer


class EnvironmentSimulator:
    """Simulate environment dynamics from an initial state.

    Uses a ``WorldModelTransformer`` to predict future latent frames
    autoregressively, then decodes them into pixel space.

    Args:
        world_model: The autoregressive world model.
        decoder: Optional decoder to convert latents to pixels.
        device: Target device.
    """

    def __init__(
        self,
        world_model: Optional[WorldModelTransformer] = None,
        decoder: Optional[nn.Module] = None,
        frame_dim: int = 512,
        device: str = "auto",
    ) -> None:
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu"
        )
        self.world_model = (world_model or WorldModelTransformer(frame_dim=frame_dim)).to(self.device).eval()
        self.decoder = decoder
        self.frame_dim = frame_dim

    @torch.no_grad()
    def simulate(
        self,
        initial_state: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        num_steps: int = 32,
        num_context: int = 4,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Run a simulation rollout.

        Args:
            initial_state: Context frames ``(1, T0, frame_dim)``. Random if None.
            actions: Actions per step ``(1, num_steps, action_dim)``.
            num_steps: Number of future frames to predict.
            num_context: Number of random context frames if *initial_state* is None.
            seed: Random seed.

        Returns:
            Predicted latent frames ``(1, T0 + num_steps, frame_dim)``.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if initial_state is None:
            initial_state = torch.randn(1, num_context, self.frame_dim, device=self.device)
        else:
            initial_state = initial_state.to(self.device)

        if actions is not None:
            actions = actions.to(self.device)

        trajectory = self.world_model.rollout(initial_state, num_steps, actions=actions)

        if self.decoder is not None:
            trajectory = self.decoder(trajectory)

        return trajectory

    @torch.no_grad()
    def simulate_interactive(
        self,
        initial_state: torch.Tensor,
        action_sequence: List[torch.Tensor],
    ) -> torch.Tensor:
        """Step-by-step interactive simulation where actions arrive one at a time."""
        state = initial_state.to(self.device)

        for action in action_sequence:
            action = action.to(self.device)
            if action.ndim == 1:
                action = action.unsqueeze(0).unsqueeze(0)
            elif action.ndim == 2:
                action = action.unsqueeze(0)
            state = self.world_model.rollout(state, num_steps=1, actions=action)

        return state
