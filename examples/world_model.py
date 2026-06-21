"""Example: World model simulation with FlashVideo.

Demonstrates action-conditioned environment simulation using the
world model transformer and dynamics model.

Usage:
    python examples/world_model.py --steps 32
"""

from __future__ import annotations

import argparse

import torch

from flashvideo.models.architectures.world_model import WorldModelTransformer
from flashvideo.world_models.dynamics import DynamicsModel
from flashvideo.world_models.simulator import EnvironmentSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashVideo — World Model Simulation")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== World Model Simulation Demo ===\n")

    # 1. Autoregressive world model
    print("1. World Model Rollout")
    world_model = WorldModelTransformer(
        frame_dim=128,
        hidden_size=256,
        depth=4,
        num_heads=4,
        action_dim=16,
        max_frames=64,
    )
    print(f"   Parameters: {sum(p.numel() for p in world_model.parameters()):,}")

    torch.manual_seed(args.seed)
    initial = torch.randn(1, args.context, 128)
    actions = torch.randn(1, args.steps, 16)

    trajectory = world_model.rollout(initial, num_steps=args.steps, actions=actions)
    print(f"   Generated trajectory: {trajectory.shape}")
    print(f"   Total frames: {trajectory.shape[1]} (context={args.context} + generated={args.steps})")

    # 2. Physics-aware dynamics
    print("\n2. Physics-Aware Dynamics Model")
    dynamics = DynamicsModel(state_dim=128, action_dim=16, hidden_dim=256)
    state = torch.randn(1, 128)
    action_seq = torch.randn(1, 10, 16)
    traj = dynamics.rollout(state, action_seq)
    print(f"   Dynamics trajectory: {traj.shape}")
    physics_loss = dynamics.physics_loss(traj)
    print(f"   Physics conservation loss: {physics_loss.item():.6f}")

    # 3. Environment simulator
    print("\n3. Environment Simulator")
    sim = EnvironmentSimulator(frame_dim=128, device=args.device)
    result = sim.simulate(num_steps=args.steps, num_context=args.context, seed=args.seed)
    print(f"   Simulation result: {result.shape}")

    print("\nDone!")


if __name__ == "__main__":
    main()
