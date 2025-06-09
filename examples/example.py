import jax.numpy as jnp
import jax
import time

from octax.environments import create_environment, print_metadata
from octax.rendering import create_video

if __name__ == "__main__":
    env, metadata = create_environment("deep")

    print_metadata(metadata)


    def policy(rng: jax.random.PRNGKey, observation: jnp.ndarray):
        return jax.random.randint(rng, (), 0, env.num_actions)


    @jax.jit
    def rollout(rng):
        def env_step(carry, _):
            rng, state, observation = carry
            rng, rng_action, rng_reset = jax.random.split(rng, 3)
            action = policy(rng_action, observation)
            next_state, next_observation, reward, terminated, truncated, info = env.step(state, action)

            next_state, next_observation, info = jax.lax.cond(terminated, env.reset,
                                                              lambda _: (next_state, next_observation, info), rng_reset)

            return (rng, next_state, next_observation), next_state

        rng, rng_reset = jax.random.split(rng)
        state, observation, info = env.reset(rng_reset)
        return jax.lax.scan(env_step, (rng, state, observation), length=10000)


    rng = jax.random.PRNGKey(0)

    # Measure compilation time
    start_compile = time.time()
    compiled = jax.block_until_ready(rollout.lower(rng).compile())
    end_compile = time.time()

    print("Compilation time (s):", end_compile - start_compile)

    # Measure execution time
    start_exec = time.time()
    final_state, states = jax.block_until_ready(compiled(rng))
    end_exec = time.time()

    print("Execution time (s):", end_exec - start_exec)

    create_video(states, display=True)