#!/bin/bash

# Run measure_time.py with different num_envs values, 1, 2, 4, ... 
# From 1 to 2^MAX_EXPONENT

echo "Starting benchmark runs with different num_envs values..."
MAX_EXPONENT=10  

for i in $(seq 0 $MAX_EXPONENT); do
    num_envs=$((2**i))
    echo "[RUN] num_envs=$num_envs (2^$i)"
    outfilename="times_$num_envs.json"
    python ablations/measure_time.py --num_envs $num_envs --output_file $outfilename
    echo "[DONE] num_envs=$num_envs (2^$i)"
    echo "[---]"
done

echo "All benchmark runs completed!"
echo "Results saved to times.json"
