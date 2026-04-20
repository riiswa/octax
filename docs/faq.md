# Frequently Asked Questions

## Installation

### JAX says it cannot find a GPU / is running on CPU

JAX defaults to CPU if a GPU-compatible build is not installed. Install the CUDA-enabled JAX separately after installing Octax:

```bash
pip install --upgrade "jax[cuda12]"
```

Adjust the CUDA version to match your driver (`cuda11`, `cuda12`). Verify with:

```python
import jax
print(jax.devices())  # should show CudaDevice(id=0)
```

### I get `ImportError` for `pygame` / `cv2` / `gymnax`

These are optional dependencies. Install the relevant extra:

```bash
pip install "octax[gui]"       # pygame, opencv, pillow
pip install "octax[training]"  # gymnax, rejax, optax, chex, hydra
pip install "octax[all]"       # everything
```

---

## Environments

### How do I know which key index maps to which action?

Each game's `action_set` lists the CHIP-8 key indices in order. Action `0` presses `action_set[0]`, action `1` presses `action_set[1]`, and so on. The final action (index `num_actions - 1`) is always a no-op.

```python
env, _ = create_environment("brix")
print(env.action_set)   # [4, 6]  →  action 0 = left, action 1 = right, action 2 = no-op
```

### The environment seems frozen / stuck in a menu

Some games require specific key presses during startup that are not part of the gameplay action set. These are handled by `startup_instructions` or `custom_startup` in the environment module. If you are loading a ROM directly (bypassing `create_environment`), run the appropriate number of warm-up instructions manually:

```python
from octax.env import run_n_instruction
state = run_n_instruction(state, 500)
```

### Why does the reward always stay at 0?

Two common causes:

1. **Wrong score register** — Use `play.py` to identify the correct register (look for BCD markers `🎯`).
2. **Startup not completed** — If the game is still on a title/menu screen, the score register may not update yet. Increase `startup_instructions`.

### Can I change the episode length?

Yes, pass `max_num_steps_per_episodes` to `create_environment`:

```python
env, _ = create_environment("brix", max_num_steps_per_episodes=10_000)
```

Or call `env.from_minutes(2.5)` to set length in real-world minutes of gameplay.

---

## Performance

### How do I run multiple environments in parallel?

Use `jax.vmap` over your reset/step functions or use the Gymnax wrapper which already handles vectorisation:

```python
import jax
from octax.environments import create_environment
from octax.wrappers import OctaxGymnaxWrapper

env, _ = create_environment("brix")
gymnax_env = OctaxGymnaxWrapper(env)
params = gymnax_env.default_params

rng = jax.random.PRNGKey(0)
rngs = jax.random.split(rng, 512)

# Vectorised reset
obs, states = jax.vmap(gymnax_env.reset_env, in_axes=(0, None))(rngs, params)
```

### First call is slow — is that normal?

Yes. JAX traces and compiles the computation graph on the first call. Subsequent calls at the same shape run the compiled kernel and are much faster. Separate compilation time from execution time:

```python
import time, jax

compiled = jax.jit(rollout).lower(rng).compile()   # explicit compilation
t0 = time.time()
jax.block_until_ready(compiled(rng))
print("Execution:", time.time() - t0)
```

### The GPU runs out of memory with many environments

Each environment carries a full CHIP-8 state (4 KB memory, 64×32 display, etc.). At 8192 parallel instances this is a few hundred MB. If you run out of memory:

- Reduce `num_envs`
- Use `disable_delay=True` (removes timer state from the trace)
- Reduce `frame_skip` to shrink observation tensors

---

## Training

### Which algorithm should I start with?

PPO (`PPOOctax`) is a solid default. It is well-understood, relatively insensitive to hyperparameters, and produces good results on most Octax games. PQN (`PQNOctax`) can be more sample-efficient on games with sparse rewards.

### How do I reproduce the paper results?

The training configuration used for all benchmark runs is in `conf/config.yaml`. Run:

```bash
python train.py env=brix agent=PPO num_seeds=12
```

Results are saved to `results/<env>/`.

### What observation preprocessing should I use?

Octax returns raw binary displays — no preprocessing is applied by default. The built-in CNN agents in `octax/agents/` use a convolutional feature extractor that works directly on the raw pixels. If you are implementing your own network, a simple stack of `Conv → ReLU` layers followed by a dense head is sufficient.

---

## Emulator

### What is modern mode vs legacy mode?

CHIP-8 has two slightly different behaviours for two instructions:

| Instruction | Legacy | Modern |
|---|---|---|
| `8XY6` / `8XYE` (shift) | Shifts `VY`, stores in `VX` | Shifts `VX` in place |
| `FX55` / `FX65` (store/load) | Increments `I` by `X+1` | `I` unchanged |
| `BXNN` (jump with offset) | Jump to `NNN + V0` | Jump to `NN + VX` |

Most games on Octax use `modern_mode=True` (the default). If a game behaves incorrectly (garbled display, infinite loop), try toggling the mode.

### Can I run SUPER-CHIP / CHIP-48 ROMs?

Octax's emulator implements the original CHIP-8 instruction set. It runs many CHIP-48 ROMs (which are largely compatible), but does **not** implement the SUPER-CHIP extended instructions (`00FE`, `00FF`, scroll instructions, 16-pixel sprites). ROMs that require these instructions will not work correctly.

### Does Octax support sound?

The emulator tracks the sound timer register, but no audio is generated. The timer is used by some games for visual timing (buzzer as a delay mechanism) even when the original sound is not needed for RL.
