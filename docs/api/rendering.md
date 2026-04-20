# Rendering API

```{eval-rst}
.. automodule:: octax.rendering
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```

## `chip8_display_to_rgb`

```python
octax.chip8_display_to_rgb(
    display: ndarray,                          # bool[64, 32]
    scale: int = 8,
    on_color: tuple[int, int, int] = (0, 255, 0),
    off_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray                                # uint8[height, width, 3]
```

Converts a CHIP-8 display buffer to an upscaled RGB image using nearest-neighbour interpolation.

## `create_color_scheme`

```python
octax.create_color_scheme(
    scheme: str = "octax",
) -> tuple[tuple[int,int,int], tuple[int,int,int]]
```

Returns `(on_color, off_color)` RGB tuples for the named palette. Available names: `octax`, `classic`, `amber`, `white`, `blue`, `retro`. Raises `ValueError` for unknown names.

## `batch_render`

```python
octax.batch_render(
    displays: ndarray,          # bool[batch, 64, 32]
    scale: int = 4,
    color_scheme: str = "octax",
) -> np.ndarray                 # uint8[grid_h, grid_w, 4]  (RGBA)
```

Renders multiple displays in a grid layout with transparent padding between cells. The output has an alpha channel; transparent regions correspond to unused grid cells when `batch_size` is not a perfect square.

```python
import matplotlib.pyplot as plt
from octax import batch_render

# states.display has shape (N, 64, 32) after a vectorised rollout
grid = batch_render(states.display, scale=4, color_scheme="retro")
plt.imshow(grid)
plt.axis("off")
plt.show()
```

## `create_video`

```python
octax.rendering.create_video(
    state: EmulatorState,         # .display must have shape (N, 64, 32)
    filename: str | None = None,
    fps: float = 60.0,
    scale: int = 8,
    color_scheme: str = "octax",
    persistence: bool = True,     # phosphor simulation
    display: bool = False,        # show in interactive window
) -> None
```

Saves and / or displays a video from a trajectory of emulator states.

- When `persistence=True`, pixels fade smoothly between frames, simulating a phosphor CRT display.
- When `display=True`, an OpenCV window opens. Press `space` to pause/resume, `q` or `ESC` to quit.
- When `filename` is provided, the video is written as an MP4.

```python
from octax.environments import create_environment
from octax.rendering import create_video
import jax

env, _ = create_environment("brix")

@jax.jit
def rollout(rng):
    def step(carry, _):
        rng, state, obs = carry
        rng, k = jax.random.split(rng)
        action = jax.random.randint(k, (), 0, env.num_actions)
        next_state, next_obs, *_ = env.step(state, action)
        return (rng, next_state, next_obs), next_state

    rng, k = jax.random.split(rng)
    state, obs, _ = env.reset(k)
    _, states = jax.lax.scan(step, (rng, state, obs), length=500)
    return states

states = rollout(jax.random.PRNGKey(0))
create_video(states, filename="brix_demo.mp4", display=True)
```
