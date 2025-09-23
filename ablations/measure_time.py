import jax
import time
import numpy as np
from octax.environments import create_environment, print_metadata
import timeit
import argparse
import json


def time_it_measure(bench, repeat=10, number=3) -> np.ndarray:
    """Measure execution time using timeit with multiple repeats."""
    times = timeit.repeat(bench, repeat=repeat, number=number)
    avg_time = np.array(times) / number
    return avg_time


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Benchmark JAX environment performance")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of parallel environments (default: 1)")
    parser.add_argument("--output_file", type=str, default="times.json",
                        help="Output file for results (default: times.json)")
    args = parser.parse_args()

    # Create environment
    env, metadata = create_environment("pong")


    @jax.jit
    def rollout(rng, state, observation):
        """Perform a rollout of 1000 steps in the environment."""

        def env_step(carry, _):
            rng, state, observation = carry
            action = 0  # Fixed action for benchmark
            next_state, next_observation, reward, terminated, truncated, info = env.step(state, action)
            return (rng, next_state, next_observation), next_state

        # Scan through 1000 environment steps
        return jax.lax.scan(env_step, (rng, state, observation), length=1000)


    # Setup random keys
    rng = jax.random.PRNGKey(0)
    num_envs = args.num_envs
    rngs = jax.random.split(rng, num_envs)

    # Reset environments outside of rollout
    reset_rngs = jax.random.split(rng, num_envs)
    states, observations, infos = jax.vmap(env.reset)(reset_rngs)

    # Measure compilation time
    print(f"Compiling for {num_envs} parallel environments...")
    start_compile = time.perf_counter()
    compiled = jax.block_until_ready(jax.jit(jax.vmap(rollout)).lower(rngs, states, observations).compile())
    end_compile = time.perf_counter()
    print(f"Compilation time: {end_compile - start_compile:.4f}s")


    # Measure execution time with more rigorous sampling
    def bench():
        final_state, rollout_states = jax.block_until_ready(compiled(rngs, states, observations))
        return final_state, rollout_states


    # Run benchmark with 50 independent samples
    print("Running benchmark (50 samples)...")
    times = time_it_measure(bench, repeat=50, number=1)

    # Calculate key statistics
    median_time = np.median(times)
    q05_time = np.percentile(times, 5)
    q95_time = np.percentile(times, 95)
    total_steps = 1000 * num_envs  # 1000 steps per rollout * number of parallel environments

    # Convert to steps per second
    median_steps_per_sec = total_steps / median_time
    q05_steps_per_sec = total_steps / q95_time  # Note: inverted because time is in denominator
    q95_steps_per_sec = total_steps / q05_time  # Note: inverted because time is in denominator

    # Prepare results for saving
    results = {
        "num_envs": num_envs,
        "total_steps": total_steps,
        "median_steps_per_sec": float(median_steps_per_sec),
        "q05_steps_per_sec": float(q05_steps_per_sec),
        "q95_steps_per_sec": float(q95_steps_per_sec),
        "median_time": float(median_time),
        "q05_time": float(q05_time),
        "q95_time": float(q95_time),
        "n_samples": len(times),
        "compilation_time": end_compile - start_compile
    }

    # Save results to JSONL file
    with open(args.output_file, "a") as f:
        f.write(json.dumps(results) + "\n")

    # Print results
    print(f"\n=== BENCHMARK RESULTS (n={len(times)}) ===")
    print(f"Number of environments: {num_envs}")
    print(f"Steps per rollout: 1000")
    print(f"Total steps per iteration: {total_steps:,}")
    print(f"\n=== TIMING STATISTICS ===")
    print(f"Median time (s): {median_time:.6f}")
    print(f"5th percentile time (s): {q05_time:.6f}")
    print(f"95th percentile time (s): {q95_time:.6f}")
    print(f"\n=== PERFORMANCE METRICS ===")
    print(f"Median steps/sec: {median_steps_per_sec:,.0f}")
    print(f"95% range: [{q05_steps_per_sec:,.0f}, {q95_steps_per_sec:,.0f}] steps/sec")
    print(f"\nResults saved to: {args.output_file}")