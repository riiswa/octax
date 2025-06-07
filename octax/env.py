import dataclasses
import time
from functools import partial
from typing import Callable, Any, Dict

import jax.numpy as jnp
import jax
import cv2
import numpy as np

from octax import create_state, fetch, execute, EmulatorState, PROGRAM_START


class OctaxEnvState(EmulatorState):
    time: int = 0
    previous_score: float = 0.
    current_score: float = 0.


def asdict_non_recursive(obj: Any) -> Dict[str, Any]:
    return {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}


class OctaxEnv:
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
        return self.instruction_frequency // self.fps

    def from_minutes(self, minutes: float):
        """Update num_steps based on desired gameplay duration in minutes."""
        self.max_num_steps_per_episodes = int(minutes * 60 * self.fps) // self.frame_skip

    def pad_frame(self, display: jnp.ndarray):
        return jnp.vstack((jnp.zeros((self.frame_skip - 1, *display.shape), dtype=display.dtype), display[None, :]))

    @partial(jax.jit, static_argnums=0)
    def reset(self, rng: jax.random.PRNGKey):
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

        observation = states.display[self.instructions_per_step - 1 :: self.instructions_per_step]

        previous_score = final_state.previous_score
        current_score = self.score_fn(final_state) * 1.0

        reward = (current_score - previous_score) * 1.0

        final_state = final_state.replace(
            current_score=current_score,
            previous_score=current_score,
            time= final_state.time + 1
        )


        return final_state, observation, reward, self.terminated_fn(final_state), final_state.time >= self.max_num_steps_per_episodes, {"score": final_state.current_score}

    @property
    def num_actions(self) -> int:
        return len(self.action_set) + 1