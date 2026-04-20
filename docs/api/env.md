# Environment API

## `OctaxEnv`

```{eval-rst}
.. automodule:: octax.env
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```

### Constructor

```python
OctaxEnv(
    rom_path: str,
    max_num_steps_per_episodes: int = 4500,
    instruction_frequency: int = 700,
    fps: int = 60,
    frame_skip: int = 4,
    action_set=None,
    score_fn: Callable = lambda _: 0.0,
    terminated_fn: Callable = lambda _: False,
    startup_instructions: int = 0,
    custom_startup: Callable | None = None,
    render_mode: str | None = None,
    disable_delay: bool = True,
    render_scale: int = 8,
    color_scheme: str = "classic",
)
```

### Properties

| Property | Type | Description |
|---|---|---|
| `num_actions` | `int` | `len(action_set) + 1` — includes no-op |
| `instructions_per_step` | `int` | `instruction_frequency // fps` |

### Methods

#### `reset`

```python
env.reset(rng: PRNGKey) -> tuple[OctaxEnvState, ndarray, dict]
```

Returns `(state, observation, info)`.

- `observation` shape: `(frame_skip, 32, 64)` — boolean array
- `info["score"]` — initial score (typically `0.0`)

#### `step`

```python
env.step(
    state: OctaxEnvState,
    action: int | ndarray,
) -> tuple[OctaxEnvState, ndarray, float, bool, bool, dict]
```

Returns `(next_state, observation, reward, terminated, truncated, info)`.

- `observation`: frame-stacked display, shape `(frame_skip, 32, 64)`
- `reward`: score delta since previous step
- `terminated`: `True` if `terminated_fn` fired
- `truncated`: `True` if `state.time >= max_num_steps_per_episodes`
- `info["score"]`: cumulative score

#### `render`

```python
env.render(state: OctaxEnvState) -> np.ndarray | None
```

Returns an RGB array of shape `(height * render_scale, width * render_scale, 3)` when `render_mode="rgb_array"`, otherwise `None`.

#### `from_minutes`

```python
env.from_minutes(minutes: float) -> None
```

Sets `max_num_steps_per_episodes` to the number of steps corresponding to `minutes` of real-time gameplay.

---

## `OctaxEnvState`

`OctaxEnvState` extends `EmulatorState` with three additional fields:

| Field | Type | Description |
|---|---|---|
| `time` | `int` | Current timestep within the episode |
| `previous_score` | `float` | Score at the previous step (for reward calculation) |
| `current_score` | `float` | Score at the current step |

---

## `create_environment`

```python
from octax.environments import create_environment

env, metadata = create_environment(
    env_id: str,
    render_mode: str | None = None,
    render_scale: int = 8,
    color_scheme: str = "classic",
    **kwargs,                    # forwarded to OctaxEnv
) -> tuple[OctaxEnv, dict]
```

Factory that imports the environment module matching `env_id`, resolves the ROM path, and constructs an `OctaxEnv`. Level-suffixed IDs (e.g. `"cavern3"`, `"space_flight7"`) are handled automatically.

**Raises** `FileNotFoundError` if the ROM cannot be found.

---

## `print_metadata`

```python
from octax.environments import print_metadata

print_metadata(metadata: dict) -> None
```

Pretty-prints a game's metadata dictionary, including title, authors, description, ROM file name, platform compatibility, controls, and CPU speed.
