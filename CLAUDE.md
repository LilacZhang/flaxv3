# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flaxv3 is a simplified implementation of DreamerV3 (world model-based reinforcement learning) using JAX and Flax NNX instead of Ninjax. It trains agents to play Atari games by learning a world model and training a policy entirely through imagined rollouts.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train an agent (uses Hydra for config overrides)
python train.py training.env_name=Boxing training.seed=1 training.device=0

# Evaluate a trained agent
python eval.py eval.env_name=Boxing eval.ckpt_path=outputs/.../ckpt/100000 eval.device=0
```

There is no test suite, linter, or build system configured.

## Architecture

The system has three main components that interact during training:

### World Model (`networks/dreamerv3.py`)
- **Encoder**: CNN (4 conv layers, progressive channel expansion) converts 64x64 RGB images to latent features
- **RSSM** (`networks/rssm.py`): Recurrent State Space Model predicts environment dynamics using a MiniGRU core, with stochastic (32 groups x 32 classes categorical) and deterministic (512-dim) state components. Separate prior (imagination) and posterior (observation) distributions
- **Decoder**: Transposed convolutions reconstruct images from latent features
- **Reward/Termination heads**: MLPs predicting rewards (SymLogTwoHot, 255 bins) and episode termination

### Actor-Critic Policy (`networks/agent.py`)
- Actor outputs discrete action distributions; Critic outputs value estimates with slow EMA target network
- Trained on imagined trajectories from the world model (no real environment gradients)
- Uses lambda returns, percentile-normalized advantages, and policy entropy regularization

### Training Loop (`train.py`)
1. Collect real environment interactions into replay buffer
2. Sample batches and update world model (reconstruction + reward + termination + KL losses)
3. Imagine future trajectories using learned policy inside the world model
4. Update actor-critic on imagined rollouts

## Key Patterns

- **Flax NNX API**: Uses the newer `flax.nnx` module (not legacy `flax.linen`). Models are `nnx.Module` subclasses; JIT via `@nnx.jit`
- **Stop gradients**: Extensive use of `sg()` (aliased `jax.lax.stop_gradient`) to control gradient flow between world model and policy
- **Hydra config**: All hyperparameters in `config/config.yaml`, overridable via CLI (e.g., `training.seed=42`)
- **JAX tree operations**: Heavy use of `jax.tree.*` for batched operations on nested state structures

## Module Layout

- `networks/net.py` — Shared building blocks: MLP heads, MiniGRU, conv/deconv layers with configurable activation and normalization
- `utils/functional.py` — Loss functions and distributions (SymLog, TwoHot, KL balancing, lambda returns)
- `utils/replaybuffer.py` — Experience replay with optional prioritized sampling
- `utils/env_wrappers.py` — Gymnasium environment builders and wrappers
- `utils/atari.py` — Atari-specific environment interface (observation preprocessing, action repeat)
- `utils/optim.py` — Optimizer setup (separate optimizers for world model and actor-critic)

## Configuration

`config/config.yaml` has four sections: `training` (env, batch sizes, train ratio), `eval` (checkpoint path, episodes), `dreamerv3` (encoder/rssm/decoder architecture), `agent` (actor-critic hyperparameters). Outputs (logs, checkpoints) go to `outputs/`.
