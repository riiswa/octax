"""
Project: OCTAX - Accelerated CHIP-8 Arcade RL Environments for JAX.
Author:
Date: 2025-09-23
Copyright: MIT License - see LICENSE file for details.
Details: Please cite the project if you use this module in your research.

"""

# Check envpool installation
try:
    import envpool

    print(envpool.__version__)
except ImportError:
    print(
        "envpool not installed. Please install it with `pip install -r requirements_ablations.txt`"
    )
    exit(1)
finally:
    print("envpool installed")

import gym
import numpy as np
import argparse
import timeit
import json
from packaging import version
from metrics_utils import compute_metrics, Metrics

is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")


def time_it_measure(bench, repeat=10, number=3) -> np.ndarray:
    """Measure execution time using timeit with multiple repeats."""
    times = timeit.repeat(bench, repeat=repeat, number=number)
    avg_time = np.array(times) / number
    return avg_time


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark EnvPool environment performance"
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
        default="Pong-v5",
        help="Name of the environment to benchmark (default: Pong-v5)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
        help="Number of steps to run (default: 1000)",
    )
    parser.add_argument(
        "--run_async",
        action="store_true",
        default=False,
        help="Whether to run the environment asynchronously (default: False)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for async mode (default: num_envs for sync, num_envs//2 for async)",
    )
    args = parser.parse_args()
    env_name = args.env_name
    num_envs = args.num_envs
    num_steps = args.num_steps
    run_async = args.run_async
    batch_size = args.batch_size

    if batch_size is None:
        batch_size = num_envs if not run_async else max(1, num_envs // 2)

    # Create environment outside benchmark for timing it fairly
    if run_async:
        env = envpool.make_gym(
            env_name,
            num_envs=num_envs,
            batch_size=batch_size,  # This is for async mode only.
            num_threads=0,  # based on the doc, let envpool decide.
            thread_affinity_offset=0,
        )
        print(f"Running in ASYNC mode: {num_envs} envs, batch_size={batch_size}")
    else:
        # For sync mode, use all environments.
        env = envpool.make_gym(env_name, num_envs=num_envs)
        print(f"Running in SYNC mode: {num_envs} envs")

    def bench():
        if run_async:
            env.async_reset()
            action = np.array([env.action_space.sample() for _ in range(batch_size)])
            for _i in range(num_steps):
                info = env.recv()[-1]
                env.send(action, info["env_id"])

        else:
            if is_legacy_gym:
                _obs = env.reset()
            else:
                _obs, _ = env.reset()

            # Generate actions for all environments
            actions = np.zeros(num_envs, dtype=np.int32)

            # Run the environment for num_steps steps
            for i in range(num_steps):
                if is_legacy_gym:
                    _obs, _rew, _done, info = env.step(actions)
                else:
                    _obs, _rew, _term, _trunc, info = env.step(actions)

    print("Running benchmark 50 samples...")
    times = time_it_measure(bench, repeat=50, number=1)
    env.close()

    # Calculate key statistics
    metrics: Metrics = compute_metrics(times, num_steps, num_envs)

    results = metrics.to_dict()

    # Save results to JSONL file
    with open(args.output_file, "a") as f:
        f.write(json.dumps(results) + "\n")

    print(f"\nResults saved to: {args.output_file}")
