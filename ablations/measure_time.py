import os

# Configure JAX memory allocation BEFORE importing jax
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import jax
import time
import numpy as np
from octax.environments import create_environment
import timeit
import argparse
import json

from gpu_memory_profiler import (
    get_gpu_memory_usage,
    get_gpu_memory_total,
    GPUMemoryMonitor,
)


def time_it_measure(bench, repeat=10, number=3) -> np.ndarray:
    """Measure execution time using timeit with multiple repeats."""
    times = timeit.repeat(bench, repeat=repeat, number=number)
    avg_time = np.array(times) / number
    return avg_time


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark JAX environment performance"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="times.json",
        help="Output file for results (default: times.json)",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="pong",
        help="Name of the environment to benchmark (default: pong)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
        help="Number of steps to run (default: 1000)",
    )
    parser.add_argument(
        "--measure_memory",
        type=bool,
        default=False,
        help="Measure memory usage (default: False)",
    )
    args = parser.parse_args()

    # Create environment
    env, metadata = create_environment(args.env_name)

    @jax.jit
    def rollout(rng, state, observation):
        """Perform a rollout of n steps in the environment."""

        def env_step(carry, _):
            rng, state, observation = carry
            action = 0  # Fixed action for benchmark
            next_state, next_observation, reward, terminated, truncated, info = (
                env.step(state, action)
            )
            return (rng, next_state, next_observation), next_state

        # Scan through n environment steps
        return jax.lax.scan(env_step, (rng, state, observation), length=args.num_steps)

    # Get baseline memory usage
    if args.measure_memory:
        baseline_memory = get_gpu_memory_usage()
        total_memory = get_gpu_memory_total()
        print(f"Baseline GPU memory usage: {baseline_memory}MB / {total_memory}MB")

    # Setup random keys
    rng = jax.random.PRNGKey(0)
    num_envs = args.num_envs
    rngs = jax.random.split(rng, num_envs)

    # Reset environments outside of rollout
    reset_rngs = jax.random.split(rng, num_envs)
    states, observations, infos = jax.vmap(env.reset)(reset_rngs)

    if args.measure_memory:
        # Check memory after environment setup
        post_env_memory = get_gpu_memory_usage()

    # Measure compilation time with memory monitoring
    print(f"Compiling for {num_envs} parallel environments...")
    if args.measure_memory:
        compile_monitor = GPUMemoryMonitor(
            interval=0.1
        )  # Sample every 100ms during compilation
        compile_monitor.start()

    start_compile = time.perf_counter()
    compiled = jax.block_until_ready(
        jax.jit(jax.vmap(rollout)).lower(rngs, states, observations).compile()
    )
    end_compile = time.perf_counter()

    if args.measure_memory:
        compile_memory_stats = compile_monitor.stop()
        post_compile_memory = get_gpu_memory_usage()
    print(f"Compilation time: {end_compile - start_compile:.4f}s")

    # Measure execution time with memory monitoring
    def bench():
        final_state, rollout_states = jax.block_until_ready(
            compiled(rngs, states, observations)
        )
        return final_state, rollout_states

    # Run a single execution with memory monitoring first
    if args.measure_memory:
        execution_monitor = GPUMemoryMonitor(
            interval=0.01
        )  # Sample every 10ms during execution
        execution_monitor.start()

    bench()  # Single execution to measure peak memory

    if args.measure_memory:
        execution_memory_stats = execution_monitor.stop()
        peak_execution_memory = get_gpu_memory_usage()

    # Run benchmark with 50 independent samples
    print(
        f"Running benchmark ({'1 sample' if args.measure_memory else '50 samples'})..."
    )
    if args.measure_memory:
        times = time_it_measure(bench, repeat=1, number=1)
    else:
        times = time_it_measure(bench, repeat=50, number=1)

    # Calculate key statistics
    median_time = np.median(times)
    q05_time = np.percentile(times, 5)
    q95_time = np.percentile(times, 95)
    total_steps = (
        args.num_steps * num_envs
    )  # num_steps steps per rollout * number of parallel environments

    # Convert to steps per second
    median_steps_per_sec = total_steps / median_time
    q05_steps_per_sec = (
        total_steps / q95_time
    )  # Note: inverted because time is in denominator
    q95_steps_per_sec = (
        total_steps / q05_time
    )  # Note: inverted because time is in denominator

    # Prepare results for saving
    results = {
        "num_envs": num_envs,
        "num_steps": args.num_steps,
        "total_steps": total_steps,
        "median_steps_per_sec": float(median_steps_per_sec),
        "q05_steps_per_sec": float(q05_steps_per_sec),
        "q95_steps_per_sec": float(q95_steps_per_sec),
        "median_time": float(median_time),
        "q05_time": float(q05_time),
        "q95_time": float(q95_time),
        "n_samples": len(times),
        "compilation_time": end_compile - start_compile,
        "memory_stats": {
            "baseline_memory_mb": baseline_memory,
            "total_gpu_memory_mb": total_memory,
            "post_env_memory_mb": post_env_memory,
            "post_compile_memory_mb": post_compile_memory,
            "peak_execution_memory_mb": execution_memory_stats["peak_memory_mb"]
            if execution_memory_stats
            else None,
            "peak_compile_memory_mb": compile_memory_stats["peak_memory_mb"]
            if compile_memory_stats
            else None,
            "env_memory_delta_mb": post_env_memory - baseline_memory
            if baseline_memory and post_env_memory
            else None,
            "compile_memory_delta_mb": post_compile_memory - post_env_memory
            if post_env_memory and post_compile_memory
            else None,
            "execution_memory_delta_mb": execution_memory_stats["peak_memory_mb"]
            - post_compile_memory
            if execution_memory_stats and post_compile_memory
            else None,
        } if args.measure_memory else None,
    }

    # Save results to JSONL file
    with open(args.output_file, "a") as f:
        f.write(json.dumps(results) + "\n")

    print(f"\nResults saved to: {args.output_file}")
