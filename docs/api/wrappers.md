# Wrappers API

## Gymnax Wrapper

`OctaxGymnaxWrapper` makes any `OctaxEnv` compatible with the [Gymnax](https://github.com/RobertTLange/gymnax) interface, enabling drop-in use with Gymnax-compatible algorithms such as those in [rejax](https://github.com/keraJLi/rejax).

```{eval-rst}
.. automodule:: octax.wrappers
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```

### Quick start

```python
from octax.environments import create_environment
from octax.wrappers import OctaxGymnaxWrapper
import jax

env, metadata = create_environment("brix")
gymnax_env = OctaxGymnaxWrapper(env)
env_params = gymnax_env.default_params

rng = jax.random.PRNGKey(0)
obs, state = gymnax_env.reset(rng, env_params)

for _ in range(100):
    rng, rng_act, rng_step = jax.random.split(rng, 3)
    action = gymnax_env.action_space(env_params).sample(rng_act)
    obs, state, reward, done, info = gymnax_env.step(
        rng_step, state, action, env_params
    )
    if done:
        obs, state = gymnax_env.reset(rng, env_params)
```

### Observation layout

The Gymnax wrapper transposes the observation from `(frame_skip, 64, 32)` to `(frame_skip, 32, 64)` to match the conventional `(channels, height, width)` layout expected by most neural network implementations.

### `OctaxEnvParams`

```python
@dataclass
class OctaxEnvParams(EnvParams):
    max_steps_in_episode: int = 4500
```

Pass a custom instance to `gymnax_env.step` / `gymnax_env.reset` to override the episode length without recreating the environment.

### Spaces

| Space | Class | Description |
|---|---|---|
| Action | `gymnax.environments.spaces.Discrete(n)` | `n = env.num_actions` |
| Observation | `gymnax.environments.spaces.Box(0, 1, (frame_skip, 32, 64))` | Float32 |

### Vectorised training with rejax

```python
from octax.agents import PPOOctax
from octax.environments import create_environment
from octax.wrappers import OctaxGymnaxWrapper
from rejax.evaluate import evaluate
import jax

env, _ = create_environment("brix")
gymnax_env = OctaxGymnaxWrapper(env)
env_params = gymnax_env.default_params

agent = PPOOctax.create_agent({}, gymnax_env, env_params)
algo = PPOOctax(
    env=gymnax_env,
    env_params=env_params,
    agent=agent,
    num_envs=512,
    num_steps=32,
    total_timesteps=5_000_000,
    learning_rate=5e-4,
)

rng = jax.random.PRNGKey(0)
train_state, metrics = jax.jit(algo.train)(rng)
```
