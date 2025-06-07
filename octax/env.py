import dataclasses
from functools import partial
from typing import Callable, Any, Dict

import jax.numpy as jnp
import jax

from octax import create_state, fetch, execute, EmulatorState, PROGRAM_START


class OctaxEnvState(EmulatorState):
    """Extended emulator state with environment-specific tracking.

    Attributes:
        time: Current timestep in the episode
        previous_score: Score from the previous step (for reward calculation)
        current_score: Current score in the episode
    """
    time: int = 0
    previous_score: float = 0.
    current_score: float = 0.


def asdict_non_recursive(obj: Any) -> Dict[str, Any]:
    """Convert dataclass to dictionary without recursive conversion.

    Args:
        obj: Dataclass instance to convert

    Returns:
        Dictionary mapping field names to their values
    """
    return {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}


class OctaxEnv:
    """JAX-compatible CHIP-8 environment for reinforcement learning.

    Provides an OpenAI Gym-style interface for CHIP-8 games with JAX/JIT compilation
    for high-performance training. Supports customizable action spaces, scoring functions,
    and termination conditions.
    """

    def __init__(
            self,
            rom_path,
            max_num_steps_per_episodes: int = 4500,
            instruction_frequency: int = 700,
            fps: int = 60,
            frame_skip: int = 4,
            action_set=None,
            score_fn: Callable[[EmulatorState], float | jnp.ndarray] = lambda _: 0.,
            terminated_fn: Callable[[EmulatorState], bool | jnp.ndarray] = lambda _: False,
    ):
        """Initialize the CHIP-8 RL environment.

        Args:
            rom_path: Path to the CHIP-8 ROM file to load
            max_num_steps_per_episodes: Maximum steps before episode truncation
            instruction_frequency: CHIP-8 CPU frequency in Hz (typically 700)
            fps: Environment frame rate (typically 60)
            frame_skip: Number of frames to skip between observations
            action_set: List/array of valid CHIP-8 key indices (0-15). If None, uses all 16 keys
            score_fn: Function to extract score from emulator state
            terminated_fn: Function to detect episode termination from emulator state
        """
        self.rom_path = rom_path
        self.max_num_steps_per_episodes = max_num_steps_per_episodes
        self.instruction_frequency = instruction_frequency
        self.fps = fps
        self.frame_skip = frame_skip
        self.terminated_fn = terminated_fn
        self.score_fn = score_fn

        if action_set is None:
            action_set = range(16)
        self.action_set = jnp.array(action_set)

        with open(rom_path, 'rb') as f:
            self.rom_data = f.read()

    @property
    def instructions_per_step(self) -> int:
        """Calculate number of CHIP-8 instructions to execute per environment step.

        Returns:
            Number of instructions per step based on frequency and FPS
        """
        return self.instruction_frequency // self.fps

    def from_minutes(self, minutes: float):
        """Set episode length based on desired gameplay duration.

        Args:
            minutes: Desired episode length in real-world minutes
        """
        self.max_num_steps_per_episodes = int(minutes * 60 * self.fps) // self.frame_skip

    def pad_frame(self, display: jnp.ndarray):
        """Pad single frame to create frame-skip compatible observation.

        Creates a batch of frames with the current frame at the end and zeros
        for previous frames, enabling consistent observation shapes.

        Args:
            display: Current frame as (width, height) array

        Returns:
            Padded observation as (frame_skip, width, height) array
        """
        return jnp.vstack((jnp.zeros((self.frame_skip - 1, *display.shape), dtype=display.dtype), display[None, :]))

    @partial(jax.jit, static_argnums=0)
    def reset(self, rng: jax.random.PRNGKey):
        """Reset the environment to initial state.

        Loads the ROM, initializes emulator state, and returns the first observation.

        Args:
            rng: JAX random key for state initialization

        Returns:
            Tuple of:
                - state: Initial OctaxEnvState
                - observation: Initial padded frame observation
                - info: Dictionary with initial score
        """
        state = create_state(rng)
        rom_array = jnp.array(list(self.rom_data), dtype=jnp.uint8)
        new_memory = state.memory.at[PROGRAM_START:PROGRAM_START + len(self.rom_data)].set(rom_array)
        state = state.replace(memory=new_memory)

        state = OctaxEnvState(
            **asdict_non_recursive(state)
        )

        return state, self.pad_frame(state.display), {"score": state.current_score}

    @partial(jax.jit, static_argnums=0)
    def step(self, state: OctaxEnvState, action: int | jnp.ndarray):
        """Execute one environment step.

        Presses the specified key, runs CHIP-8 instructions for the appropriate
        duration, updates timers, calculates rewards, and checks termination.

        Args:
            state: Current environment state
            action: Action index (0 to len(action_set)-1 for keys, len(action_set) for no-op)

        Returns:
            Tuple of:
                - next_state: Updated environment state
                - observation: Frame stack observation (frame_skip, width, height)
                - reward: Scalar reward (score difference)
                - terminated: Boolean indicating if episode ended
                - truncated: Boolean indicating if episode was truncated (max steps)
                - info: Dictionary with current score
        """

        def run_instruction(state, _):
            state, instruction = fetch(state)
            state = execute(state, instruction)
            return state, state

        state = jax.lax.cond(
            action == (self.num_actions - 1),
            lambda s: s,
            lambda s: s.replace(keypad=s.keypad.at[self.action_set[action]].set(1)),
            state
        )

        final_state, states = jax.lax.scan(run_instruction, state, length=self.instructions_per_step * self.frame_skip)
        final_state: OctaxEnvState = final_state.replace(
            delay_timer=jnp.maximum(final_state.delay_timer - 1, 0),
            sound_timer=jnp.maximum(final_state.sound_timer - 1, 0),
        )

        final_state = jax.lax.cond(
            action == (self.num_actions - 1),
            lambda s: s,
            lambda s: s.replace(keypad=s.keypad.at[self.action_set[action]].set(0)),
            final_state
        )

        observation = states.display[self.instructions_per_step - 1:: self.instructions_per_step]

        previous_score = final_state.previous_score
        current_score = self.score_fn(final_state) * 1.0

        reward = (current_score - previous_score) * 1.0

        final_state = final_state.replace(
            current_score=current_score,
            previous_score=current_score,
            time=final_state.time + 1
        )

        return final_state, observation, reward, self.terminated_fn(
            final_state), final_state.time >= self.max_num_steps_per_episodes, {"score": final_state.current_score}

    @property
    def num_actions(self) -> int:
        """Get total number of available actions.

        Returns:
            Number of actions (length of action_set + 1 for no-op)
        """
        return len(self.action_set) + 1