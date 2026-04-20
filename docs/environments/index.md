# Environments

Octax ships with **22 classic CHIP-8 arcade games** across five genre categories.
All environments share the same interface and are fully interchangeable.

---

<div class="envs-grid">

<div class="env-card"><a href="games.html#airplane"><img src="../_static/imgs/airplane.gif" alt="Airplane"/><p class="env-name">Airplane</p><p class="env-category">Shooter</p></a></div>

<div class="env-card"><a href="games.html#blinky"><img src="../_static/imgs/blinky.gif" alt="Blinky"/><p class="env-name">Blinky</p><p class="env-category">Puzzle</p></a></div>

<div class="env-card"><a href="games.html#brix"><img src="../_static/imgs/brix.gif" alt="Brix"/><p class="env-name">Brix</p><p class="env-category">Action</p></a></div>

<div class="env-card"><a href="games.html#cavern"><img src="../_static/imgs/cavern1.gif" alt="Cavern"/><p class="env-name">Cavern</p><p class="env-category">Exploration · 7 levels</p></a></div>

<div class="env-card"><a href="games.html#deep8"><img src="../_static/imgs/deep.gif" alt="Deep8"/><p class="env-name">Deep8</p><p class="env-category">Shooter</p></a></div>

<div class="env-card"><a href="games.html#filter"><img src="../_static/imgs/filter.gif" alt="Filter"/><p class="env-name">Filter</p><p class="env-category">Action</p></a></div>

<div class="env-card"><a href="games.html#flight-runner"><img src="../_static/imgs/flight_runner.gif" alt="Flight Runner"/><p class="env-name">Flight Runner</p><p class="env-category">Exploration</p></a></div>

<div class="env-card"><a href="games.html#missile-command"><img src="../_static/imgs/missile.gif" alt="Missile Command"/><p class="env-name">Missile Command</p><p class="env-category">Strategy</p></a></div>

<div class="env-card"><a href="games.html#pong"><img src="../_static/imgs/pong.gif" alt="Pong"/><p class="env-name">Pong</p><p class="env-category">Action</p></a></div>

<div class="env-card"><a href="games.html#rocket"><img src="../_static/imgs/rocket.gif" alt="Rocket"/><p class="env-name">Rocket</p><p class="env-category">Strategy</p></a></div>

<div class="env-card"><a href="games.html#shooting-stars"><img src="../_static/imgs/shooting_stars.gif" alt="Shooting Stars"/><p class="env-name">Shooting Stars</p><p class="env-category">Shooter</p></a></div>

<div class="env-card"><a href="games.html#space-flight"><img src="../_static/imgs/space_flight1.gif" alt="Space Flight"/><p class="env-name">Space Flight</p><p class="env-category">Exploration · 10 levels</p></a></div>

<div class="env-card"><a href="games.html#spacejam"><img src="../_static/imgs/spacejam.gif" alt="Spacejam!"/><p class="env-name">Spacejam!</p><p class="env-category">Exploration</p></a></div>

<div class="env-card"><a href="games.html#squash"><img src="../_static/imgs/squash.gif" alt="Squash"/><p class="env-name">Squash</p><p class="env-category">Action</p></a></div>

<div class="env-card"><a href="games.html#submarine"><img src="../_static/imgs/submarine.gif" alt="Submarine"/><p class="env-name">Submarine</p><p class="env-category">Strategy</p></a></div>

<div class="env-card"><a href="games.html#tank-battle"><img src="../_static/imgs/tank.gif" alt="Tank Battle"/><p class="env-name">Tank Battle</p><p class="env-category">Strategy</p></a></div>

<div class="env-card"><a href="games.html#target-shooter"><img src="../_static/imgs/target_shooter1.gif" alt="Target Shooter"/><p class="env-name">Target Shooter</p><p class="env-category">Shooter · 3 levels</p></a></div>

<div class="env-card"><a href="games.html#tetris"><img src="../_static/imgs/tetris.gif" alt="Tetris"/><p class="env-name">Tetris</p><p class="env-category">Puzzle</p></a></div>

<div class="env-card"><a href="games.html#ufo"><img src="../_static/imgs/ufo.gif" alt="UFO"/><p class="env-name">UFO</p><p class="env-category">Strategy</p></a></div>

<div class="env-card"><a href="games.html#vertical-brix"><img src="../_static/imgs/vertical_brix.gif" alt="Vertical Brix"/><p class="env-name">Vertical Brix</p><p class="env-category">Action</p></a></div>

<div class="env-card"><a href="games.html#wipe-off"><img src="../_static/imgs/wipe_off.gif" alt="Wipe Off"/><p class="env-name">Wipe Off</p><p class="env-category">Action</p></a></div>

<div class="env-card"><a href="games.html#worm"><img src="../_static/imgs/worm.gif" alt="Worm"/><p class="env-name">Worm</p><p class="env-category">Puzzle</p></a></div>

</div>

<a class="see-more" href="games.html">See all game specifications →</a>

---

## Common Interface

### Creating an environment

```python
from octax.environments import create_environment

env, metadata = create_environment("brix")

# With rendering
env, metadata = create_environment(
    "brix",
    render_mode="rgb_array",
    render_scale=8,
    color_scheme="octax",
)
```

### Observation Space

Every environment returns `Box(False, True, (frame_skip, 32, 64), bool)`.

The `frame_skip` axis provides temporal context (default: 4) — equivalent to frame-stacking in Atari environments.

### Action Space

`Discrete(len(action_set) + 1)`. The last action is always a **no-op**. Each game automatically restricts the action space to the keys it actually uses.

### Reward

`reward_t = score(state_t) − score(state_{t−1})` — score delta per step.

### Constructor arguments

| Argument | Default | Description |
|---|---|---|
| `max_num_steps_per_episodes` | `4500` | Steps before truncation |
| `instruction_frequency` | `700` | CHIP-8 CPU speed (Hz) |
| `fps` | `60` | Frame rate |
| `frame_skip` | `4` | Stacked frames in observation |
| `disable_delay` | `False` | Disable delay/sound timers |
| `render_mode` | `None` | `"rgb_array"` or `None` |
| `render_scale` | `8` | Pixel upscale factor |
| `color_scheme` | `"classic"` | Colour palette |

```{toctree}
:hidden:

games
```
