from typing import Tuple, Dict, Any
import jax
import jax.numpy as jnp

from flax.struct import dataclass
from gymnax.environments.environment import Environment, EnvParams
from gymnax.environments.spaces import Discrete, Box

from octax.env import OctaxEnv, OctaxEnvState


@dataclass
class OctaxEnvParams(EnvParams):
    """Gymnax-compatible parameters for Octax environment."""
    max_steps_in_episode: int = 4500


class OctaxGymnaxWrapper(Environment[OctaxEnvState, OctaxEnvParams]):
    """Gymnax wrapper for OctaxEnv."""

    def __init__(self, octax_env: OctaxEnv):
        """Initialize wrapper with an OctaxEnv instance.

        Args:
            octax_env: Configured OctaxEnv instance
        """
        self._octax_env = octax_env

    @property
    def default_params(self) -> OctaxEnvParams:
        """Return default environment parameters."""
        return OctaxEnvParams(
            max_steps_in_episode=self._octax_env.max_num_steps_per_episodes
        )

    def step_env(
            self,
            key: jax.Array,
            state: OctaxEnvState,
            action: int,
            params: OctaxEnvParams,
    ) -> Tuple[jax.Array, OctaxEnvState, jax.Array, jax.Array, Dict[Any, Any]]:
        """Execute one environment step."""
        next_state, obs, reward, terminated, truncated, info = self._octax_env.step(state, action)

        done = terminated | truncated

        return obs.transpose(0, 2, 1), next_state, reward, done, info

    def reset_env(
            self,
            key: jax.Array,
            params: OctaxEnvParams
    ) -> Tuple[jax.Array, OctaxEnvState]:
        """Reset environment to initial state."""
        state, obs, info = self._octax_env.reset(key)
        return obs.transpose(0, 2, 1), state

    @property
    def name(self) -> str:
        """Environment name."""
        return f"Octax_{self._octax_env.rom_path.split('/')[-1].split('.')[0]}"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self._octax_env.num_actions

    def action_space(self, params: OctaxEnvParams) -> Discrete:
        """Action space of the environment."""
        return Discrete(self.num_actions)

    def observation_space(self, params: OctaxEnvParams) -> Box:
        """Observation space of the environment."""
        # CHIP-8 display is 64x32, with frame_skip stacking
        height, width = 32, 64
        frame_skip = self._octax_env.frame_skip

        return Box(
            low=0.0,
            high=1.0,
            shape=(frame_skip, height, width),
            dtype=jnp.float32
        )
