# Octax: Accelerated CHIP-8 Arcade Environments for JAX

**A high-performance JAX-based implementation of classic arcade environments for reinforcement learning research.**

## Overview

This repository implements a comprehensive suite of classic arcade games as reinforcement learning environments, using authentic CHIP-8 emulation with JAX for unprecedented training speed through JIT compilation and vectorization. The system enables researchers to train RL agents on iconic games like Pong, Breakout (Brix), Tetris, and many others with massive parallelization.

### Key Research Contributions

- **Vectorized RL Environments**: JAX-native implementation enabling thousands of parallel game instances
- **Authentic Retro Gaming**: Faithful CHIP-8 emulation supporting both legacy and modern interpreter modes
- **Performance Benchmarking**: Comprehensive comparison with existing RL environment frameworks (EnvPool, Gymnasium)
- **Automatic Game Analysis**: Tools for discovering game mechanics and creating new environments
- **Research-Ready Interface**: Drop-in compatibility with popular RL frameworks via Gymnasium wrapper

## Research Significance

Classic arcade games provide excellent RL benchmarks due to their:
- **Clear reward structures** and well-defined objectives
- **Diverse gameplay mechanics** requiring different strategies
- **Computational efficiency** enabling large-scale experiments
- **Interpretable action spaces** and visual observations
- **Historical significance** as established benchmarks

Our JAX implementation achieves **10-100x speedup** over traditional environments, enabling researchers to run experiments that were previously computationally prohibitive.

## Repository Structure

### Core Implementation
- `octax/` - Main library with emulator, environments, and utilities
- `octax/agents/ppo.py` - PPO implementation for Octax
- `octax/environments/` - 20+ classic games as RL environments (Pong, Brix, Tetris, etc.)

### Research Examples
- `train.py` - Multi-seed training with Hydra configuration

### Benchmarking and Analysis
- `ablations/` - Performance comparison studies
  - `measure_time.py` - Octax performance benchmarking
  - `measure_envpool.py` - Comparison with EnvPool framework
  - `gpu_memory_profiler.py` - Memory usage analysis
  - `run_measure_time.sh` - Automated benchmarking suite
- `conf/` - Hyperparameter sweeps and configuration management

### Game Analysis Tools
- `main.py` - Interactive emulator with score detection for analyzing games
- `roms/` - Game files and custom environment examples

### Research Infrastructure
- `tests/` - Comprehensive test suite for emulator correctness
- `tutorial/` - Documentation for researchers and developers
- `requirements_*.txt` - Environment dependencies for reproducibility

## Quick Start

```python
import jax
from octax.environments import create_environment

# Create vectorized environment
env, metadata = create_environment("brix")  # Breakout clone

# JAX-native training loop
@jax.jit
def training_step(rng, state, obs):
    # Your RL algorithm here
    action = policy(obs)
    next_state, next_obs, reward, done, truncated, info = env.step(state, action)
    return next_state, next_obs, reward

# Massively parallel execution
rng = jax.random.PRNGKey(0)
batch_size = 4096  # 4K parallel environments
rngs = jax.random.split(rng, batch_size)
states, observations, _ = jax.vmap(env.reset)(rngs)

# Single compiled function handles all environments
final_states, rewards = jax.vmap(training_step)(rngs, states, observations)
```

## Available Games

The repository includes 20+ carefully analyzed classic games:
- **Pong** - Classic paddle game, ideal for learning basics
- **Brix** - Breakout clone with multiple lives system  
- **Tetris** - Strategic piece-placement requiring planning
- **Deep8** - Complex action game with scoring mechanics
- **And many more** - Each with documented reward structures

## Reproducibility

All experiments are reproducible with provided configuration files:
- `conf/config.yaml` - Single-game training setup
- `conf/sweep.yaml` - Multi-game hyperparameter sweeps
- Fixed seeds and deterministic JAX compilation
- Docker containers available for exact environment reproduction

## Installation

```bash
pip install -r requirements.txt

# For GPU acceleration (recommended)
pip install --upgrade "jax[cuda12]"

# For training experiments  
pip install -r requirements_training.txt
```

*This repository contains the complete research artifact for our ICLR 2026 submission, including all code, data, and experimental configurations needed to reproduce our results.*