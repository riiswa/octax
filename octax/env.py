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
            terminated_fn: Callable[[EmulatorState], bool] = lambda _: False,
            score_fn: Callable[[EmulatorState], float] = lambda _: 0.,
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

        state = state.replace(keypad=state.keypad.at[self.action_set[action]].set(1))

        final_state, states = jax.lax.scan(run_instruction, state, length=self.instructions_per_step * self.frame_skip)
        final_state: OctaxEnvState = final_state.replace(
            delay_timer=jnp.maximum(final_state.delay_timer - 1, 0),
            sound_timer=jnp.maximum(final_state.sound_timer - 1, 0),
        )

        final_state = final_state.replace(keypad=final_state.keypad.at[self.action_set[action]].set(0))

        observation = states.display[self.instructions_per_step - 1 :: self.instructions_per_step]

        final_state = final_state.replace(current_score=self.score_fn(final_state) * 1.0)

        reward = (final_state.current_score - final_state.previous_score) * 1.0

        final_state = final_state.replace(
            previous_score=final_state.current_score,
            time= final_state.time + 1
        )


        return final_state, observation, reward, self.terminated_fn(final_state), final_state.time >= self.max_num_steps_per_episodes, {"score": final_state.current_score}

    @property
    def num_actions(self) -> int:
        return len(self.action_set)

if __name__ == "__main__":
    env = OctaxEnv("../c8games/TETRIS")

    def policy(rng: jax.random.PRNGKey, observation: jnp.ndarray):
        return jax.random.randint(rng, (), 0, env.num_actions)

    @jax.jit
    def rollout(rng):
        def env_step(carry, _):
            rng, state, observation = carry
            rng, rng_action = jax.random.split(rng)
            action = policy(rng_action, observation)
            next_state, next_observation, reward, terminated, truncated, info = env.step(state, action)
            return (rng, next_state, next_observation), observation

        state, observation, info = env.reset(rng)
        return jax.lax.scan(env_step, (rng, state, observation), length=env.max_num_steps_per_episodes)


    rng = jax.random.PRNGKey(0)


    # Measure compilation time
    start_compile = time.time()
    compiled = jax.block_until_ready(rollout.lower(rng).compile())
    end_compile = time.time()

    print("Compilation time (s):", end_compile - start_compile)

    # Measure execution time
    start_exec = time.time()
    final_state, observations = jax.block_until_ready(compiled(rng))
    end_exec = time.time()

    print("Execution time (s):", end_exec - start_exec)


    frames = np.array(observations[:, -1]) * 255  # Convert to 0-255 range
    frames = frames.astype(np.uint8)

    # Display frames
    for i, frame in enumerate(frames):
        # Resize for visibility
        display_frame = cv2.resize(frame.T, (640, 320),
                                   interpolation=cv2.INTER_NEAREST)

        cv2.imshow('CHIP-8 Display', display_frame)

        # Exit on 'q' or ESC, pause on spacebar
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord(' '):  # Spacebar to pause
            cv2.waitKey(0)

    cv2.destroyAllWindows()