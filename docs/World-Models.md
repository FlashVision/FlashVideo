# World Models

FlashVideo includes a world model subsystem inspired by **NVIDIA Cosmos** and **GAIA-1** for physics-aware environment simulation and action-conditioned video prediction.

## Overview

World models learn to simulate environments by predicting future states given past observations and actions. They are fundamental to:

- **Robotics** — Planning in latent space before acting
- **Autonomous Driving** — Simulating traffic scenarios
- **Gaming** — Procedural environment generation
- **Physical AI** — Understanding physics from video

## Components

### World Model Transformer

Autoregressive causal transformer that predicts future latent frames:

```python
from flashvideo.models.architectures.world_model import WorldModelTransformer

model = WorldModelTransformer(
    frame_dim=512,
    hidden_size=768,
    depth=12,
    num_heads=12,
    action_dim=64,
    max_frames=64,
)

# Predict future from context
trajectory = model.rollout(initial_frames, num_steps=32, actions=actions)
```

### Physics-Aware Dynamics

State-transition model with energy conservation regularisation:

```python
from flashvideo.world_models.dynamics import DynamicsModel

dynamics = DynamicsModel(state_dim=512, action_dim=64)
trajectory = dynamics.rollout(initial_state, action_sequence)
physics_loss = dynamics.physics_loss(trajectory)
```

### Action-Conditioned Generation

Generate video frames conditioned on agent actions:

```python
from flashvideo.world_models.action_conditioned import ActionConditionedGenerator

gen = ActionConditionedGenerator(state_dim=512, action_dim=64)
trajectory = gen.generate_trajectory(initial_state, actions)
```

### Long-Horizon Memory

Compressed memory bank for simulations beyond the transformer's context window:

```python
from flashvideo.world_models.memory import LongHorizonMemory

memory_module = LongHorizonMemory(dim=768, mem_size=128)
memory = memory_module.init_memory(batch_size=1, device=device)
augmented, memory = memory_module(hidden_states, memory)
```

## High-Level API

```python
from flashvideo import SceneSimulator

sim = SceneSimulator()
sim.simulate(
    prompt="a robot arm picking up a red block",
    actions=["reach", "grasp", "lift"],
    output="simulation.mp4",
    num_frames=64,
)
```

## Training

```bash
flashvideo train --config configs/flashvideo_world_model.yaml
```

## Key Design Decisions

1. **Causal masking** — Autoregressive prediction prevents information leakage
2. **Action gating** — Actions modulate hidden states via learned gates
3. **Physics prior** — Energy conservation loss encourages physically-plausible dynamics
4. **Compressed memory** — Fixed-size memory bank with learned compression for long horizons
