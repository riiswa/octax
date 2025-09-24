#!/bin/bash

# Project: OCTAX - Accelerated CHIP-8 Arcade RL Environments for JAX.
# Author:
# Date: 2025-09-23
# Copyright: MIT License - see LICENSE file for details.
# Details: Please cite the project if you use this module in your research.

# Run measure_time.py with different num_envs values, 1, 2, 4, ... 
# From 1 to 2^MAX_EXPONENT

echo "Starting benchmark runs with different num_envs values..."
START_EXPONENT=13
MAX_EXPONENT=15
NUM_STEPS=100


export XLA_PYTHON_CLIENT_PREALLOCATE=true # JAX preallocation 

function run_benchmark {
    for i in $(seq $START_EXPONENT $MAX_EXPONENT); do
        num_envs=$((2**i))
        echo "[RUN] num_envs=$num_envs (2^$i)"
        outfilename="times_octax.num_envs=$num_envs.num_steps=$NUM_STEPS.json"
        python ablations/measure_time.py --num_envs $num_envs --output_file $outfilename --num_steps $NUM_STEPS
        echo "[DONE] num_envs=$num_envs (2^$i)"
        echo "[---]"
    done
}

function run_benchmark_mem_check {
    export XLA_PYTHON_CLIENT_PREALLOCATE=false # JAX preallocation
    for i in $(seq $START_EXPONENT $MAX_EXPONENT); do
        num_envs=$((2**i))
        echo "[RUN] num_envs=$num_envs (2^$i)"
        outfilename="times_octax_mem_check.num_steps=$NUM_STEPS.json"
        python ablations/measure_time.py --num_envs $num_envs --output_file $outfilename --num_steps $NUM_STEPS --measure_memory=true
        echo "[DONE] num_envs=$num_envs (2^$i)"
        echo "[---]"
    done

}

function run_benchmark_envpool_sync {
    for i in $(seq $START_EXPONENT $MAX_EXPONENT); do
        num_envs=$((2**i))
        echo "[RUN] num_envs=$num_envs (2^$i)"
        outfilename="times_envpool.num_envs=$num_envs.num_steps=$NUM_STEPS.json"
        python ablations/measure_envpool.py --num_envs $num_envs --output_file $outfilename --num_steps $NUM_STEPS
        echo "[DONE] num_envs=$num_envs (2^$i)"
        echo "[---]"
    done
}

function run_benchmark_envpool_async {
    for i in $(seq $START_EXPONENT $MAX_EXPONENT); do
        num_envs=$((2**i))
        echo "[RUN] num_envs=$num_envs (2^$i)"
        outfilename="times_envpool.num_envs=$num_envs.num_steps=$NUM_STEPS.json"
        python ablations/measure_envpool.py --num_envs $num_envs --output_file $outfilename --num_steps $NUM_STEPS --run_async
        echo "[DONE] num_envs=$num_envs (2^$i)"
        echo "[---]"
    done
}

function run_benchmark_once {
    NUM_ENVS=1350
    python ablations/measure_time.py --num_envs $NUM_ENVS --output_file "times_$NUM_ENVS.json"
}



function cleanup {
    unset XLA_PYTHON_CLIENT_PREALLOCATE
}

# ==== OCTAX ===== (uncomment to run)
# run_benchmark
run_benchmark_mem_check
# run_benchmark_once

# ==== ENVPOOL ===== (uncomment to run)
# run_benchmark_envpool_sync

cleanup

echo "All benchmark runs completed!"
echo "Results saved to times.json"
