"""Physics-aware dynamics model.

Learns the transition dynamics of an environment, predicting the next
state given the current state and action.  Includes an optional physics
prior that enforces conservation laws and smooth trajectories.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsPrior(nn.Module):
    """Soft physics prior encouraging energy conservation and smooth motion."""

    def __init__(self, state_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.energy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def compute_energy(self, state: torch.Tensor) -> torch.Tensor:
        return self.energy_net(state)

    def conservation_loss(self, states: torch.Tensor) -> torch.Tensor:
        """Penalise large energy changes between consecutive states."""
        energies = self.energy_net(states)
        diffs = (energies[:, 1:] - energies[:, :-1]).pow(2)
        return diffs.mean()


class DynamicsModel(nn.Module):
    """State-transition dynamics model.

    Predicts ``s_{t+1} = f(s_t, a_t)`` and optionally applies a physics
    prior for more physically-plausible predictions.

    Args:
        state_dim: State embedding dimension.
        action_dim: Action dimension.
        hidden_dim: MLP hidden size.
        use_physics_prior: Enable energy-conservation regularisation.
    """

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 64,
        hidden_dim: int = 512,
        use_physics_prior: bool = True,
    ) -> None:
        super().__init__()
        input_dim = state_dim + action_dim
        self.transition = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.residual_gate = nn.Sequential(nn.Linear(state_dim, state_dim), nn.Sigmoid())
        self.physics = PhysicsPrior(state_dim) if use_physics_prior else None

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: Current state ``(B, state_dim)``.
            action: Applied action ``(B, action_dim)``.
            target: Optional ground-truth next state for loss computation.

        Returns:
            Dictionary with ``"next_state"`` and optionally ``"loss"``.
        """
        x = torch.cat([state, action], dim=-1)
        delta = self.transition(x)
        gate = self.residual_gate(delta)
        next_state = state + gate * delta

        result: Dict[str, torch.Tensor] = {"next_state": next_state}

        if target is not None:
            result["loss"] = F.mse_loss(next_state, target)

        return result

    def rollout(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a trajectory of states given a sequence of actions.

        Args:
            initial_state: ``(B, state_dim)``.
            actions: ``(B, T, action_dim)``.

        Returns:
            States ``(B, T+1, state_dim)`` including the initial state.
        """
        states = [initial_state]
        state = initial_state
        for t in range(actions.shape[1]):
            out = self.forward(state, actions[:, t])
            state = out["next_state"]
            states.append(state)
        trajectory = torch.stack(states, dim=1)

        result = {"trajectory": trajectory}
        if self.physics is not None:
            result["physics_loss"] = self.physics.conservation_loss(trajectory)
        return trajectory

    def physics_loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Compute physics conservation loss over a trajectory."""
        if self.physics is None:
            return torch.tensor(0.0, device=trajectory.device)
        return self.physics.conservation_loss(trajectory)
