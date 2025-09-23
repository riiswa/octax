# Check envpool installation
try:
    import envpool
    print(envpool.__version__)
except ImportError:
    print("envpool not installed. Please install it with `pip install -r requirements_ablations.txt`")
    exit(1)
finally:
    print("envpool installed")

import numpy as np
import argparse
    
    
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Benchmark EnvPool environment performance")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of parallel environments (default: 1)")
    parser.add_argument("--output_file", type=str, default="times.json",
                        help="Output file for results (default: times.json)")
    parser.add_argument("--env_name", type=str, default="Pong-v5", 
                        help="Name of the environment to benchmark (default: Pong-v5)")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum number of steps to run (default: 1000)")
    args = parser.parse_args()


    # Run the pong env with envpool
    env = envpool.make(args.env_name, env_type="gym", num_envs=args.num_envs)

    # Run the env for max_steps steps..
    for i in range(args.max_steps):
        act = np.zeros(args.num_envs, dtype=int)
        obs, rew, term, trunc, info = env.step(act)

    env.close()

    print("Done")