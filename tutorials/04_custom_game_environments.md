# Creating Custom Game Environments

Octax supports adding any CHIP-8 game as an RL environment. The process has three stages: playing the game to understand its mechanics, identifying the right registers, and writing a short Python module. This tutorial walks through all three.

## Understanding Game Environment Structure

Every Octax environment is a small Python module inside `octax/environments/`. Let's read one of the simplest to understand the pattern:

```python
# octax/environments/missile.py

from octax import EmulatorState

rom_file = "Missile [David Winter].ch8"

def score_fn(state: EmulatorState) -> float:
    return state.V[7]          # Score lives in register V7

def terminated_fn(state: EmulatorState) -> bool:
    return state.V[6] == 0     # Game over when V6 (missiles) reaches 0

action_set = [8]               # Only key 8 (fire) is used

startup_instructions = 50      # Skip past the title screen

metadata = {
    "title": "Missile Command",
    "release": "1996",
    "authors": ["David Winter"],
    ...
}
```

That is the complete environment definition. Everything else — step logic, observation stacking, reward calculation, JIT compilation — is handled automatically by `OctaxEnv`.

## Stage 1 — Play the Game

The most reliable way to understand a CHIP-8 game is to run it in Octax's interactive emulator (`play.py`). This displays all 16 registers in real time and prints a `🎯 BCD!` marker whenever a BCD-conversion instruction fires, which is the strongest indicator of a score register.

```bash
python play.py roms/YourGame.ch8
```

**Controls in the emulator:**
- `D` — toggle the debug overlay (register values, BCD markers)
- `P` — pause / unpause
- `R` — reset
- `+` / `-` — increase / decrease CPU speed
- `ESC` — quit

While playing, look for:

| Signal | What it means |
|---|---|
| BCD marker `🎯` on register `Vx` | `Vx` holds the displayed score |
| Register that starts high and decreases | Likely a lives / ammo counter |
| Register that only increases | Likely a score or progress counter |
| Register that snaps to `0` or `255` at game over | Likely a game-state flag |

## Stage 2 — Identify the Key Registers

Open the debug overlay (`D`) and play until you score points, lose a life, and reach game over. Note down:

1. **Score register** — which `Vx` tracks your score
2. **Termination register** — which `Vx` signals game over and what value it takes
3. **Used keys** — which keys you actually pressed (shown in the overlay)
4. **Startup skip** — how many instructions pass before the game loop begins (skip menus / splash screens)

### Tip: Check the `.8o` Source

If a decompiled source file exists in `roms/`, it provides the definitive register map. Look for:

```
# register comments at the top of the file
:alias score v7
:alias lives v6
```

### Tip: Using `print_all_registers`

Run `play.py` for 30 seconds and wait for the automatic register dump. The registers marked with `📈` (monotonically increasing) are strong score candidates; `📉` (decreasing) are likely countdown timers.

## Stage 3 — Write the Environment Module

Create a new file in `octax/environments/`:

```python
# octax/environments/my_game.py

from octax import EmulatorState

# ── 1. ROM filename ──────────────────────────────────────────────────────────
rom_file = "MyGame.ch8"   # must exist in roms/

# ── 2. Score function ────────────────────────────────────────────────────────
def score_fn(state: EmulatorState) -> float:
    """Return the current score from the emulator state."""
    return state.V[5]          # replace with your score register

# ── 3. Termination function ──────────────────────────────────────────────────
def terminated_fn(state: EmulatorState) -> bool:
    """Return True when the game is definitively over."""
    return state.V[12] == 0    # replace with your termination condition

# ── 4. Action set ────────────────────────────────────────────────────────────
action_set = [4, 6]            # CHIP-8 key indices actually used by the game

# ── 5. Startup skip (optional) ───────────────────────────────────────────────
startup_instructions = 500     # instructions to run before the first reset

# ── 6. Metadata ──────────────────────────────────────────────────────────────
metadata = {
    "title": "My Game",
    "description": "Brief description of the game.",
    "release": "2024",
    "authors": ["Your Name"],
    "roms": {
        "sha1hashofrom": {
            "file": "MyGame.ch8",
            "platforms": ["originalChip8"],
        }
    },
}
```

Place the ROM in `roms/MyGame.ch8` and test immediately:

```python
from octax.environments import create_environment

env, metadata = create_environment("my_game")

import jax
rng = jax.random.PRNGKey(0)
state, obs, info = env.reset(rng)
print("Reset OK, obs shape:", obs.shape)

for _ in range(100):
    rng, k = jax.random.split(rng)
    action = jax.random.randint(k, (), 0, env.num_actions)
    state, obs, reward, terminated, truncated, info = env.step(state, action)
    if terminated or truncated:
        print("Episode ended. Score:", info["score"])
        break
```

## Advanced Patterns

### Multi-component Score

Some games store their score across multiple registers (BCD digits, high/low bytes, etc.):

```python
import jax.numpy as jnp

def score_fn(state: EmulatorState) -> float:
    # Score stored as two BCD digits: V[0] = tens, V[1] = ones
    return state.V[0] * 10.0 + state.V[1]
```

### Custom Startup Sequence

If the game requires specific key presses during initialisation (e.g. "press 0 to start"), use `custom_startup`:

```python
from octax.env import run_n_instruction

def custom_startup(state: EmulatorState) -> EmulatorState:
    # Press key 0 to dismiss the title screen
    state = state.replace(keypad=state.keypad.at[0].set(1))
    state = run_n_instruction(state, 200)
    state = state.replace(keypad=state.keypad.at[0].set(0))
    state = run_n_instruction(state, 50)
    return state
```

Set `custom_startup` in your module and omit `startup_instructions`:

```python
# my_game.py
startup_instructions = 0   # handled by custom_startup above
```

### Protecting Against Score Overflow

CHIP-8 registers are 8-bit (0–255). Games that would overflow often wrap to 0. Clamp the score to avoid spurious negative rewards:

```python
import jax

def score_fn(state: EmulatorState) -> float:
    return jax.lax.cond(
        state.V[0] > 200,
        lambda: jnp.astype(0, jnp.uint8),
        lambda: state.V[0],
    )
```

### Levelled Games

Cavern and Space Flight ship multiple ROMs (one per level). The `create_environment` function handles the naming convention automatically:

```python
# File naming: <env_id><level>.ch8  →  env_id = "cavern", module = cavern.py
env, _ = create_environment("cavern3")   # loads cavern3.ch8
env, _ = create_environment("space_flight5")  # loads space_flight5.ch8
```

To support this in your own module, simply leave `rom_file` pointing to the base ROM; `create_environment` will override it when a level suffix is detected.

## Testing Your Environment

Octax's test suite does not cover custom environments automatically, but you can quickly write a sanity-check:

```python
# tests/test_my_game.py

import jax
import jax.numpy as jnp
from octax.environments import create_environment

def test_my_game_resets():
    env, _ = create_environment("my_game")
    rng = jax.random.PRNGKey(0)
    state, obs, info = env.reset(rng)
    assert obs.shape == (env.frame_skip, 32, 64)
    assert info["score"] >= 0

def test_my_game_steps():
    env, _ = create_environment("my_game")
    rng = jax.random.PRNGKey(1)
    state, obs, info = env.reset(rng)
    for _ in range(50):
        rng, k = jax.random.split(rng)
        action = jax.random.randint(k, (), 0, env.num_actions)
        state, obs, reward, terminated, truncated, info = env.step(state, action)

def test_episode_terminates():
    """Episode must terminate within max_steps."""
    env, _ = create_environment("my_game", max_num_steps_per_episodes=500)
    rng = jax.random.PRNGKey(42)
    state, obs, _ = env.reset(rng)
    done = False
    for _ in range(600):
        rng, k = jax.random.split(rng)
        action = jax.random.randint(k, (), 0, env.num_actions)
        state, obs, _, terminated, truncated, _ = env.step(state, action)
        if terminated or truncated:
            done = True
            break
    assert done, "Episode never terminated"
```

## Submitting Your Environment

If you want to contribute your environment back to Octax, see the [Contributing Guide](../contributing.md) for the full pull-request checklist. The short version:

1. Add your module to `octax/environments/`
2. Add the ROM to `roms/` (verify it is public domain or appropriately licensed)
3. Add an entry to `docs/environments/games.md`
4. Write a test in `tests/`
5. Open a pull request with a short description of the game's mechanics
