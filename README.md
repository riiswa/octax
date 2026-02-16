# <img src="https://github.com/riiswa/octax/blob/main/imgs/octax_logo.png?raw=true" alt="Octax Logo" width="80" align="left"/> OCTAX: Accelerated CHIP-8 Arcade Environments for Reinforcement Learning in JAX


<p>   <strong>High-performance CHIP-8 arcade game environments for reinforcement learning research</strong> </p> <p align="center">   <a href="#key-features">Features</a> •   <a href="#installation">Installation</a> •   <a href="#quick-start">Quick Start</a> •   <a href="#available-games">Games</a> •   <a href="#performance">Performance</a> •   <a href="#citation">Citation</a> </p>

📄 Preprint is available at: [https://arxiv.org/abs/2510.01764](https://arxiv.org/abs/2510.01764)

------

OCTAX provides a JAX-based suite of classic arcade game environments implemented through CHIP-8 emulation. It offers **orders-of-magnitude speedups** over traditional CPU emulators while maintaining perfect fidelity to original game mechanics, making it ideal for large-scale reinforcement learning experimentation.

<p align="center">   <img src="https://github.com/riiswa/octax/blob/main/imgs/figure1_games.png?raw=true" alt="OCTAX Games Overview" width="800"/>   <br>   <em>Sample of 20+ classic arcade games available in OCTAX</em> </p>

## Why OCTAX?

Modern RL research demands extensive experimentation with thousands of parallel environments and comprehensive hyperparameter sweeps. Traditional arcade emulators remain CPU-bound, creating a computational bottleneck. OCTAX solves this by:

- **🚀 GPU Acceleration**: End-to-end JAX implementation runs thousands of game instances in parallel
- **⚡ Massive Speedups**: 14× faster than CPU-based alternatives at high parallelization
- **🎮 Authentic Games**: Perfect fidelity to original CHIP-8 mechanics across 20+ games
- **🔧 Easy Integration**: Compatible with Gymnasium and popular RL frameworks
- **📊 Research-Ready**: Spans puzzle, action, strategy, and exploration genres

## Key Features

### End-to-End GPU Acceleration

- Fully vectorized CHIP-8 emulation in JAX
- JIT-compiled for maximum performance
- Scales from single environments to 8192+ parallel instances
- Eliminates CPU-GPU transfer bottlenecks

### Diverse Game Portfolio

- 20+ Games

   spanning multiple genres:

  - **Puzzle**: Tetris, Blinky (Pac-Man), Worm (Snake)
  - **Action**: Brix (Breakout), Pong, Squash, Wipe-Off
  - **Strategy**: Missile Command, Tank Battle, UFO
  - **Exploration**: Cavern (7 levels), Space Flight (10 levels)
  - **Shooter**: Airplane, Deep8, Shooting Stars

### Research-Friendly Design

- Gymnax-compatible interface for seamless JAX integration
- Customizable reward functions and termination conditions
- Frame stacking and observation preprocessing
- Multiple color schemes for visualization
- Modular architecture for easy extension

### Built for Scale

- Train experiments that took days in hours
- Run comprehensive hyperparameter sweeps feasibly
- Achieve statistical validity with hundreds of seeds
- Perfect for curriculum learning and meta-RL research

## Installation

```bash
pip install octax
```

For GPU acceleration (highly recommended):

```bash
pip install --upgrade "jax[cuda12]"

# Optional to run the train.py script
pip install -r requirements_training.txt
```

From source:

```bash
git clone https://github.com/riiswa/octax.git
cd octax
pip install -e .
```

## Quick Start

### Basic Environment Usage

```python
import jax
import jax.numpy as jnp
from octax.environments import create_environment

# Create environment
env, metadata = create_environment("brix")
print(f"Playing: {metadata['title']}")

# Simple random policy
@jax.jit
def rollout(rng):
    state, obs, info = env.reset(rng)
    
    def step(carry, _):
        rng, state, obs = carry
        rng, action_rng = jax.random.split(rng)
        action = jax.random.randint(action_rng, (), 0, env.num_actions)
        
        next_state, next_obs, reward, terminated, truncated, info = env.step(state, action)
        return (rng, next_state, next_obs), reward
    
    final_carry, rewards = jax.lax.scan(step, (rng, state, obs), length=1000)
    return jnp.sum(rewards)

# Run episode
rng = jax.random.PRNGKey(0)
total_reward = rollout(rng)
print(f"Total reward: {total_reward}")
```

### Vectorized Training (64 Parallel Environments)

```python
# Run 64 environments in parallel
@jax.jit
def vectorized_rollout(rng, num_envs=64):
    rngs = jax.random.split(rng, num_envs)
    return jax.vmap(rollout)(rngs)

rewards = vectorized_rollout(rng, 64)
print(f"Mean reward: {jnp.mean(rewards):.2f} ± {jnp.std(rewards):.2f}")
```

### Gymnax Integration

```python
from octax.environments import create_environment
from octax.wrappers import OctaxGymnaxWrapper

# Create Gymnax-compatible environment
env, metadata = create_environment("brix")
gymnax_env = OctaxGymnaxWrapper(env)
env_params = gymnax_env.default_params

# Use with any Gymnax-compatible algorithm
rng = jax.random.PRNGKey(0)
obs, state = gymnax_env.reset(rng, env_params)

for _ in range(100):
    rng, rng_action, rng_step = jax.random.split(rng, 3)
    action = gymnax_env.action_space(env_params).sample(rng_action)
    obs, state, reward, done, info = gymnax_env.step(
        rng_step, state, action, env_params
    )
```

**Observation Space**: The agent receives the raw CHIP-8 display as a `(frame_skip, 32, 64)` boolean array, where `frame_skip` (default: 4) provides temporal information. Each frame is a 32×64 monochrome image capturing the complete visual state—exactly what a human player would see.

**Action Space**: A discrete space where actions map to game-specific CHIP-8 keys. For example, Pong uses `[1, 4]` (up/down), Brix uses `[4, 6]` (left/right), and Tetris uses `[4, 5, 6, 7]` (rotate/move). An additional no-op action is always available. Games automatically configure their relevant action subsets, eliminating irrelevant keys from the action space.

## Available Games

| Category        | Games                                                        | Required Capabilities                      |
| --------------- | ------------------------------------------------------------ | ------------------------------------------ |
| **Puzzle**      | Tetris, Blinky, Worm                                         | Long-horizon planning, spatial reasoning   |
| **Action**      | Brix, Pong, Squash, Vertical Brix, Wipe Off, Filter          | Timing, prediction, reactive control       |
| **Strategy**    | Missile Command, Rocket, Submarine, Tank Battle, UFO         | Resource management, tactical decisions    |
| **Exploration** | Cavern (7 levels), Flight Runner, Space Flight (10 levels), Spacejam! | Spatial exploration, continuous navigation |
| **Shooter**     | Airplane, Deep8, Shooting Stars                              | Simple reaction, basic timing              |

All environments support:

- Customizable frame skip and action repeat
- Configurable episode lengths
- Built-in score tracking
- Multiple rendering modes
- Frame stacking for temporal information

## Performance

OCTAX achieves substantial speedups over traditional CPU-based environments through JAX vectorization:

<p align="center">   <img src="https://github.com/riiswa/octax/blob/main/imgs/figure3_performance.png?raw=true" alt="Performance Scaling" width="600"/>   <br>   <em>OCTAX vs EnvPool performance scaling across parallelization levels (RTX 3090)</em> </p>

**Key Results:**

- **14× faster** than EnvPool at 8192 parallel environments
- **350,000 steps/second** on a single RTX 3090
- **Near-linear scaling** up to GPU memory limits
- **Sub-second compilation** for fast iteration

This enables:

- **Comprehensive experiments**: Run 100+ hyperparameter configurations overnight
- **Statistical rigor**: Train with 50+ random seeds for reliable results
- **Rapid prototyping**: Iterate on algorithms with immediate feedback

## Training Results

PPO training across 16 diverse games (5M timesteps, 12 seeds each):

<p align="center">   <img src="https://github.com/riiswa/octax/blob/main/imgs/figure2_training.png?raw=true" alt="Training Results" width="800"/>   <br>   <em>PPO learning curves showing diverse challenges across game genres</em> </p>

## Project Structure

```
octax/
├── octax/                  # Core package
│   ├── emulator.py        # CHIP-8 emulator implementation
│   ├── env.py             # RL environment wrapper
│   ├── environments/      # Game-specific configurations
│   ├── instructions/      # CHIP-8 instruction handlers
│   ├── rendering.py       # Visualization utilities
│   └── wrappers.py        # Gymnax compatibility wrapper
├── examples/              # Usage examples
│   ├── example.py        # Basic rollout
│   ├── training_on_octax.py  # PPO training demo
│   └── rendering_demo.py # Visualization examples
├── tests/                 # Comprehensive test suite
└── tutorial/              # In-depth tutorials
```

## Contributing

We welcome contributions! Ways to contribute:

### Adding New Games

1. **Find a CHIP-8 ROM**: Many public domain games available

2. Analyze the game

   : Use our interactive emulator (

   ```
   main.py
   ```

   ) to identify:

   - Score registers (look for BCD operations marked with 🎯)
   - Termination conditions (game over states)
   - Required controls (which keys are used)

3. **Create environment file**: See `octax/environments/` for examples

4. **Test and submit**: Ensure score/termination work correctly

```python
# octax/environments/my_game.py
from octax import EmulatorState

rom_file = "my_game.ch8"

def score_fn(state: EmulatorState) -> float:
    return state.V[5]  # Score in register V5

def terminated_fn(state: EmulatorState) -> bool:
    return state.V[12] == 0  # Game over when V12 reaches 0

action_set = [4, 6]  # Left/Right controls
startup_instructions = 500  # Skip menu screens

metadata = {
    "title": "My Game",
    "description": "A classic arcade game",
    "release": "2024",
    "authors": ["Author Name"]
}
```

### Other Contributions

- Bug fixes and performance improvements
- Documentation enhancements
- Additional examples and tutorials
- New features (Super-CHIP8 support, etc.)

Please open an issue to discuss major changes before implementing.

## Citation

If you use OCTAX in your research, please cite our paper:

```bibtex
@misc{radji2025octax,
    title={Octax: Accelerated CHIP-8 Arcade Environments for Reinforcement Learning in JAX},
    author={Waris Radji and Thomas Michel and Hector Piteau},
    year={2025},
    eprint={2510.01764},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## License

OCTAX is released under the MIT License. See [LICENSE](https://github.com/riiswa/octax/blob/main/LICENSE) for details.

## Acknowledgments

- CHIP-8 games from the [CHIP-8 Database](https://github.com/chip-8/chip-8-database)
- Inspired by the [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
- Built with [JAX](https://github.com/google/jax), [Flax](https://github.com/google/flax), and [Optax](https://github.com/deepmind/optax)

------

<p align="center">   Made with ❤️ for the RL research community </p>
