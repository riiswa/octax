# Octax: Accelerated CHIP-8 Arcade Environments for JAX

A high-performance JAX-based implementation of CHIP-8 arcade environments designed for reinforcement learning research. Octax provides blazing-fast, JIT-compiled game environments that can run thousands of episodes in parallel.

## 🎯 Primary Goal

**Accelerated RL Training**: Enable researchers to train RL agents on classic arcade games with unprecedented speed through JAX's JIT compilation and vectorization capabilities.

## 📅 Roadmap

- [x] Core CHIP-8 emulator with JAX/JIT compilation
- [x] Complete instruction set implementation (all 35 CHIP-8 opcodes)
- [x] RL environment wrapper with OpenAI Gym-style interface
- [x] Interactive score detection tools (BCD tracking, register analysis)
- [x] Modern vs legacy mode support
- [x] Comprehensive test suite with pytest
- [ ] Implement 15 game environments (e.g. Pong, Space Invaders, Tetris, Snake, Pac-Man, Asteroids, Frogger, Missile Command)
- [ ] Implement PPO and DQN baseline agent
- [ ] Add rendering tools (video recording, episode replay, batch visualization)
- [ ] Benchmark GPU, CPU (and TPU) performance across different hardware
- [ ] Create comprehensive API documentation and tutorials
- [ ] Set up continuous integration and code coverage
- [ ] Add SuperCHIP-8 support (optional)

## 🚀 Key Features

- **Parallel Execution**: Vectorized environments for massive batch training
- **Authentic Emulation**: Faithful CHIP-8 implementation with both modern and legacy modes
- **Easy Environment Creation**: Simple framework for adding new arcade games
- **Score Detection Tools**: Built-in utilities for discovering game mechanics
- **Research-Ready**: Gymnasium-style interface optimized for JAX workflows

## Installation

```bash
pip install -r requirements.txt

# For GPU acceleration (recommended)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Quick Start: RL Training

```python
from octax.environments import create_environment, print_metadata
import jax

# Create vectorized environment
env, metadata = create_environment("brix")

print_metadata(metadata)

# JIT-compiled training loop
@jax.jit
def train_step(rng, state, observation):
    rng, action_rng = jax.random.split(rng)
    action = jax.random.randint(action_rng, (), 0, env.num_actions)

    next_state, next_obs, reward, terminated, truncated, info = env.step(state, action)

    # Reset if episode ends
    rng, reset_rng = jax.random.split(rng)
    next_state, next_obs, info = jax.lax.cond(
        terminated | truncated,
        lambda _: env.reset(reset_rng),
        lambda _: (next_state, next_obs, info),
        None
    )

    return rng, next_state, next_obs, reward


# Initialize
rng = jax.random.PRNGKey(0)
state, obs, info = env.reset(rng)

for _ in range(1000):
    rng, state, obs, reward = train_step(rng, state, obs)
```

## How to Contribute

### Adding a New Environment

Follow these steps to add support for a new CHIP-8 game:

#### 1. Analyze the Game

Use the interactive emulator to understand the game mechanics:

```bash
python main.py  # Loads BRIX by default
```

**Controls for Analysis:**
- `D`: Toggle debug overlay to see all 16 registers
- `P`: Pause to examine state
- `+/-`: Adjust speed for detailed observation

**What to Look For:**
- **Score registers**: Which registers increase when you score points?
- **Life/health registers**: Which registers decrease when you take damage?
- **Game state registers**: Which registers indicate game over conditions?
- **BCD operations**: Watch for 🎯 markers indicating score displays

#### 2. Create Environment Definition

Create a new file in `octax/environments/`:

```python
# octax/environments/my_game.py
from octax import EmulatorState

# Path to your ROM file (place in c8games/ directory)
rom_path = "../../c8games/MY_GAME"

def score_fn(state: EmulatorState):
    """Extract score from game state."""
    # Example: Score stored in register V3
    return state.V[3]

def terminated_fn(state: EmulatorState):
    """Detect if game has ended."""
    # Example: Game over when lives (V12) reach 0
    return state.V[12] == 0

# Define which keys the agent can press
# Common patterns:
action_set = [4, 6]        # Left/Right only
# action_set = [2, 4, 6, 8] # Up/Left/Right/Down
# action_set = [5]          # Single action button

metadata = # Refer to https://github.com/chip-8/chip-8-database/tree/master
```

### Score Detection Methodology

The interactive emulator (`main.py`) includes sophisticated score detection:

#### BCD Detection
- **What it is**: Binary Coded Decimal operations (FX33 instruction)
- **Why it matters**: Games use BCD to display scores on screen
- **How to use**: Look for 🎯 markers in debug output

```python
# The detector automatically identifies BCD operations
# When you see: "🎯 BCD! V5 = 156 -> MEM[0x300]"
# This means register V5 likely contains the score
```

#### Register Trend Analysis
- **Increasing trends** 📈: Likely score or progress counters
- **Decreasing trends** 📉: Likely lives, health, or timers
- **Stable values**: Probably configuration or unused registers

### Submission Guidelines

1. **ROM Requirements**: Ensure your ROM is freely distributable or provide instructions for obtaining it legally
2. **Documentation**: Include analysis of the game mechanics in your environment file
3. **Testing**: Validate that score and termination work correctly

## License

MIT License - see LICENSE file for details.

## Citation

If you use Octax in your research, please cite:

```bibtex
@software{octax2025,
  title={Octax: Accelerated CHIP-8 Arcade Environments for JAX},
  author={Waris Radji},
  year={2025},
  url={https://github.com/riiswa/octax}
}
```