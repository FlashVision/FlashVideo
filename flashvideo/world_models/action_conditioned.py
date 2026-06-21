"""Action-conditioned video generation.

Generate future video frames conditioned on agent actions (e.g. robot
commands, game inputs, driving controls).  Combines a dynamics model
with a video decoder for pixel-level prediction.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionEncoder(nn.Module):
    """Encode discrete or continuous actions into a latent space."""

    def __init__(self, action_space: int = 64, embed_dim: int = 256, continuous: bool = True) -> None:
        super().__init__()
        if continuous:
            self.encoder = nn.Sequential(
                nn.Linear(action_space, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Embedding(action_space, embed_dim),
            )
        self.continuous = continuous

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        if not self.continuous:
            return self.encoder[0](action.long())
        return self.encoder(action)


class ActionConditionedGenerator(nn.Module):
    """Generate video frames conditioned on a sequence of actions.

    Architecture:
    1. Encode initial frame(s) to a latent state.
    2. Apply action-conditioned dynamics to predict future states.
    3. Decode latent states to video frames.

    Args:
        state_dim: Latent state dimension.
        action_dim: Raw action dimension.
        action_embed_dim: Embedded action dimension.
        hidden_dim: Dynamics MLP hidden size.
        continuous_actions: Whether actions are continuous vectors or discrete ids.
    """

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 64,
        action_embed_dim: int = 256,
        hidden_dim: int = 512,
        continuous_actions: bool = True,
    ) -> None:
        super().__init__()
        self.action_encoder = ActionEncoder(action_dim, action_embed_dim, continuous_actions)

        self.transition = nn.Sequential(
            nn.Linear(state_dim + action_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        initial_states: torch.Tensor,
        actions: torch.Tensor,
        target_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            initial_states: ``(B, state_dim)`` initial latent state.
            actions: ``(B, T, action_dim)`` action sequence.
            target_states: ``(B, T, state_dim)`` ground-truth future states.

        Returns:
            Dictionary with ``"predicted_states"`` and optionally ``"loss"``.
        """
        b, t_steps, _ = actions.shape
        state = self.state_encoder(initial_states)

        predicted = []
        for t in range(t_steps):
            action_emb = self.action_encoder(actions[:, t])
            combined = torch.cat([state, action_emb], dim=-1)
            delta = self.transition(combined)
            state = state + delta
            predicted.append(state)

        predicted_states = torch.stack(predicted, dim=1)
        result: Dict[str, torch.Tensor] = {"predicted_states": predicted_states}

        if target_states is not None:
            result["loss"] = F.mse_loss(predicted_states, target_states)

        return result

    @torch.no_grad()
    def generate_trajectory(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a state trajectory from initial state and actions.

        Returns:
            ``(B, T+1, state_dim)`` including initial state.
        """
        out = self.forward(initial_state, actions)
        all_states = torch.cat([initial_state.unsqueeze(1), out["predicted_states"]], dim=1)
        return all_states
