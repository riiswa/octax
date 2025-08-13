"""
Gymnasium compatibility wrapper for Octax environments.

This module provides a wrapper that makes Octax environments compatible with the
Gymnasium API while preserving JAX performance benefits.
"""

from typing import Any, Dict, Optional, Tuple, Union
import jax
import jax.numpy as jnp
import numpy as np

import gymnasium as gym
from gymnasium import spaces

class GymnasiumWrapper(gym.Env):
    """
    Gymnasium-compatible wrapper for Octax environments.
    
    This wrapper maintains internal state and provides the standard Gymnasium API
    while leveraging JAX's performance benefits under the hood.
    
    Example:
        ```python
        from octax.environments import create_environment
        from octax.gymnasium_wrapper import GymnasiumWrapper
        
        # Create Octax environment
        octax_env, metadata = create_environment("brix")
        
        # Wrap for Gymnasium compatibility
        gym_env = GymnasiumWrapper(octax_env)
        
        # Use standard Gymnasium API
        obs, info = gym_env.reset()
        obs, reward, terminated, truncated, info = gym_env.step(action)
        ```
    """

    def __init__(self, octax_env, seed: Optional[int] = None):
        """
        Initialize the Gymnasium wrapper.

        Args:
            octax_env: An Octax environment instance
            seed: Optional seed for random number generation
        """

        self.octax_env = octax_env
        self._state = None
        self._rng_key = jax.random.PRNGKey(seed if seed is not None else 42)

        # Set up Gymnasium-compatible attributes
        self.metadata = octax_env.metadata.copy()

        # Create action space (Discrete with num_actions)
        self.action_space = spaces.Discrete(octax_env.num_actions)

        # Create observation space (Box for the frame stack)
        # Observation shape is (frame_skip, height, width) from the test
        # We need to infer this from a reset to get exact dimensions
        temp_key = jax.random.PRNGKey(0)
        temp_state, temp_obs, _ = octax_env.reset(temp_key)
        obs_shape = temp_obs.shape
        self.observation_space = spaces.Box(
            low=0, high=1, shape=obs_shape, dtype=np.float32
        )

        # Environment spec (optional but good practice)
        self.spec = None  # Can be set if needed for registration

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional seed for this episode
            options: Optional environment options (unused)
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            self._rng_key = jax.random.PRNGKey(seed)

        # Split key for this reset
        reset_key, self._rng_key = jax.random.split(self._rng_key)

        # Reset the Octax environment
        self._state, observation, info = self.octax_env.reset(reset_key)

        # Convert JAX arrays to numpy for Gymnasium compatibility
        observation = np.array(observation)
        info = {k: np.array(v) if hasattr(v, 'shape') else v for k, v in info.items()}

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        # Execute step in Octax environment
        (self._state, observation, reward, terminated, 
         truncated, info) = self.octax_env.step(self._state, action)

        # Convert JAX arrays to numpy for Gymnasium compatibility
        observation = np.array(observation)
        reward = float(reward)
        terminated = bool(terminated)
        truncated = bool(truncated)
        info = {k: np.array(v) if hasattr(v, 'shape') else v for k, v in info.items()}

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Render the current environment state.
        
        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self._state is None:
            return None

        rendered = self.octax_env.render(self._state)
        return np.array(rendered) if rendered is not None else None

    def close(self):
        """Clean up environment resources."""
        # Octax environments don't need explicit cleanup
        pass

    @property 
    def unwrapped(self):
        """Access the underlying Octax environment."""
        return self.octax_env


class VectorizedGymnasiumWrapper:
    """
    Vectorized Gymnasium wrapper that can handle multiple environments in parallel.
    
    This leverages JAX's vmap for efficient batch processing while maintaining 
    Gymnasium compatibility.
    """

    def __init__(self, octax_env, num_envs: int, seed: Optional[int] = None):
        """
        Initialize vectorized wrapper.

        Args:
            octax_env: An Octax environment instance
            num_envs: Number of parallel environments
            seed: Optional seed for random number generation
        """

        self.octax_env = octax_env
        self.num_envs = num_envs
        self._states = None

        # Initialize RNG keys for each environment
        base_key = jax.random.PRNGKey(seed if seed is not None else 42)
        self._rng_keys = jax.random.split(base_key, num_envs)

        # Set up spaces (same as single environment)
        single_wrapper = GymnasiumWrapper(octax_env, seed=0)
        self.action_space = single_wrapper.action_space
        self.observation_space = single_wrapper.observation_space
        self.metadata = single_wrapper.metadata.copy()
        self.spec = None

        # Create vectorized functions
        self._vmap_reset = jax.vmap(octax_env.reset)
        self._vmap_step = jax.vmap(octax_env.step)
        self._vmap_render = jax.vmap(octax_env.render) if hasattr(octax_env, 'render') else None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset all environments."""
        if seed is not None:
            base_key = jax.random.PRNGKey(seed)
            self._rng_keys = jax.random.split(base_key, self.num_envs)

        # Split keys for reset
        reset_keys = jax.random.split(self._rng_keys[0], self.num_envs)
        self._rng_keys = jax.random.split(self._rng_keys[0], self.num_envs)

        # Vectorized reset
        self._states, observations, infos = self._vmap_reset(reset_keys)

        # Convert to numpy
        observations = np.array(observations)
        # Note: infos handling for vectorized case may need special treatment

        return observations, {}  # Simplified info for vectorized case

    def step(self, actions):
        """Step all environments with given actions."""
        if self._states is None:
            raise RuntimeError("Must call reset() before step()")

        # Vectorized step
        (self._states, observations, rewards, 
         terminated, truncated, infos) = self._vmap_step(self._states, actions)

        # Convert to numpy
        observations = np.array(observations)
        rewards = np.array(rewards)
        terminated = np.array(terminated)
        truncated = np.array(truncated)

        return observations, rewards, terminated, truncated, {}

    def render(self):
        """Render all environments."""
        if self._states is None or self._vmap_render is None:
            return None
        return np.array(self._vmap_render(self._states))

    def close(self):
        """Clean up resources."""
        pass


def make_gymnasium_env(env_id: str, **kwargs) -> GymnasiumWrapper:
    """
    Convenience function to create a Gymnasium-compatible Octax environment.
    
    Args:
        env_id: Octax environment ID (e.g., "brix", "pong")
        **kwargs: Additional arguments passed to create_environment
        
    Returns:
        GymnasiumWrapper instance
    """
    from octax.environments import create_environment
    octax_env, metadata = create_environment(env_id, **kwargs)
    return GymnasiumWrapper(octax_env)


def make_vectorized_env(env_id: str, num_envs: int, **kwargs) -> VectorizedGymnasiumWrapper:
    """
    Convenience function to create a vectorized Gymnasium-compatible environment.
    
    Args:
        env_id: Octax environment ID
        num_envs: Number of parallel environments
        **kwargs: Additional arguments passed to create_environment
        
    Returns:
        VectorizedGymnasiumWrapper instance
    """
    from octax.environments import create_environment  
    octax_env, metadata = create_environment(env_id, **kwargs)
    return VectorizedGymnasiumWrapper(octax_env, num_envs)
