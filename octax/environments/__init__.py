import importlib
from types import ModuleType
from typing import Callable

from octax import EmulatorState
import jax.numpy as jnp

from octax.env import OctaxEnv


class EnvDef(ModuleType):
    rom_path: str
    score_fn: Callable[[EmulatorState], float | jnp.ndarray]
    terminated_fn: Callable[[EmulatorState], bool | jnp.ndarray]
    action_set: list


def create_environment(env_id: str, **kwargs):
    module: EnvDef = importlib.import_module(f"octax.environments.{env_id}")
    return OctaxEnv(
        rom_path=module.rom_path,
        score_fn=module.score_fn,
        terminated_fn=module.terminated_fn,
        action_set=module.action_set,
        **kwargs
    )

if __name__ == "__main__":
    # Example
    import jax.numpy as jnp
    import jax
    import cv2
    import numpy as np
    import time

    env = create_environment("brix")

    def policy(rng: jax.random.PRNGKey, observation: jnp.ndarray):
        return jax.random.randint(rng, (), 0, env.num_actions)


    @jax.jit
    def rollout(rng):
        def env_step(carry, _):
            rng, state, observation = carry
            rng, rng_action, rng_reset = jax.random.split(rng, 3)
            action = policy(rng_action, observation)
            next_state, next_observation, reward, terminated, truncated, info = env.step(state, action)

            next_state, next_observation, info = jax.lax.cond(terminated, env.reset, lambda _: (next_state, next_observation, info), rng_reset)

            return (rng, next_state, next_observation), observation

        rng, rng_reset = jax.random.split(rng)
        state, observation, info = env.reset(rng_reset)
        return jax.lax.scan(env_step, (rng, state, observation), length=1000000)


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

        cv2.putText(display_frame, f"{i + 1}/{len(frames)}", (540, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('CHIP-8 Display', display_frame)

        # Exit on 'q' or ESC, pause on spacebar
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord(' '):  # Spacebar to pause
            cv2.waitKey(0)

    cv2.destroyAllWindows()
