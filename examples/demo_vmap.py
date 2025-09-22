import jax.numpy as jnp
import jax
import time 
import numpy as np

from octax.environments import create_environment, print_metadata
from octax.rendering import create_video

import timeit


def time_perf_counter_measure(func, *args, **kwargs):
    start = time.perf_counter()
    func(*args, **kwargs)
    end = time.perf_counter()
    return end - start

def time_it_measure(bench, repeat=10, number=3, *args) -> np.ndarray:
    times = timeit.repeat(bench, repeat=repeat, number=number)
    avg_time = np.array(times) / np.ones(len(times)) * number
    return avg_time



if __name__ == "__main__":
    env, metadata = create_environment("pong")

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
        return jax.lax.scan(env_step, (rng, state, observation), length=1000)
    
    rng = jax.random.PRNGKey(0)
    # 10000 is the number of environments
    # this is the var to fluctuate.
    rngs = jax.random.split(rng, 1000)

    # Measure compilation time
    start_compile = time.perf_counter()
    compiled = jax.block_until_ready(jax.jit(jax.vmap(rollout)).lower(rngs).compile())
    end_compile = time.perf_counter()

    print("Compilation time (s):", end_compile - start_compile)

    # Measure execution time
    def bench():
        _final_state, _states = jax.block_until_ready(compiled(rngs))

    times = time_it_measure(bench)
    times = np.array(times)
    print(times.shape)
    print(times)
    mean_time = times.mean()
    q1 = np.quantile(times, 0.25)
    q3 = np.quantile(times, 0.75)

    print("Execution times (s):", times)
    print("Mean time (s):", mean_time)
    print("Q1 (s):", q1)
    print("Q3 (s):", q3)