"""
Project: OCTAX - Accelerated CHIP-8 Arcade RL Environments for JAX.
Author:
Date: 2025-09-23
Copyright: MIT License - see LICENSE file for details.
Details: Please cite the project if you use this module in your research.
"""

import numpy as np
from pydantic import BaseModel


class Metrics(BaseModel):
    num_envs: int
    num_steps: int
    total_steps: int
    median_steps_per_sec: float
    q05_steps_per_sec: float
    q95_steps_per_sec: float
    median_time: float
    q05_time: float
    q95_time: float
    n_samples: int
    compilation_time: float | None
    memory_stats: dict | None

    def to_dict(self) -> dict:
        return self.model_dump()


def compute_metrics(times: np.ndarray, num_steps: int, num_envs: int) -> dict:
    # Calculate key statistics
    median_time = np.median(times)
    q05_time = np.percentile(times, 5)
    q95_time = np.percentile(times, 95)
    total_steps = (
        num_steps * num_envs
    )  # num_steps steps per rollout * number of parallel environments

    # Convert to steps per second
    median_steps_per_sec = total_steps / median_time
    q05_steps_per_sec = (
        total_steps / q95_time
    )  # Note: inverted because time is in denominator
    q95_steps_per_sec = (
        total_steps / q05_time
    )  # Note: inverted because time is in denominator

    return Metrics(
        num_envs=num_envs,
        num_steps=num_steps,
        total_steps=total_steps,
        median_steps_per_sec=float(median_steps_per_sec),
        q05_steps_per_sec=float(q05_steps_per_sec),
        q95_steps_per_sec=float(q95_steps_per_sec),
        median_time=float(median_time),
        q05_time=float(q05_time),
        q95_time=float(q95_time),
        n_samples=len(times),
        compilation_time=None,
        memory_stats=None,
    )
