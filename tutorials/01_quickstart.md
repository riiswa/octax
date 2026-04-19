# Octax Quickstart Guide

Octax is a JAX-based CHIP-8 emulator and reinforcement learning environment suite. This guide covers installation and basic usage.

## What is Octax?

Octax provides CHIP-8 games as reinforcement learning environments. The library implements a CHIP-8 emulator using JAX, allowing parallel execution of multiple game instances for training RL agents on classic games like Pong, Brix (Breakout), and Tetris.

Octax uses JAX's JIT compilation for performance. The first execution compiles functions to optimized code, with subsequent runs executing at high speed. This enables efficient scaling from single environments to large parallel batches.

## Installation

First, install Octax and its dependencies:

```bash
pip install octax
```

For GPU acceleration (highly recommended), install JAX with CUDA support:

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Your First Octax Environment

Let's start by running a simple CHIP-8 game. The most straightforward way to experience Octax is through its environment interface, which wraps the low-level emulator in a familiar API:

```python
import jax
import jax.numpy as jnp
from octax.environments import create_environment

# Create a Deep8 environment (a challenging puzzle game)
env, metadata = create_environment("deep")

print(f"Game: {metadata['name']}")
print(f"Actions: {env.num_actions}")
```

This creates an environment for the Deep8 puzzle game. The `metadata` tells you about the game, while `env` gives you the actual environment to interact with.

Now let's run a simple random agent:

```python
def random_policy(rng_key, observation):
    """A simple random policy - just press random buttons!"""
    return jax.random.randint(rng_key, (), 0, env.num_actions)

@jax.jit
def run_episode(rng_key):
    """Run a complete episode with random actions"""
    # Reset the environment to start fresh
    rng_reset, rng_play = jax.random.split(rng_key)
    state, observation, info = env.reset(rng_reset)
    
    total_reward = 0
    steps = 0
    
    def step_fn(carry, _):
        rng, state, obs, total_reward, steps = carry
        rng, rng_action = jax.random.split(rng)
        
        # Choose a random action
        action = random_policy(rng_action, obs)
        
        # Take the action
        next_state, next_obs, reward, terminated, truncated, info = env.step(state, action)
        
        return (rng, next_state, next_obs, total_reward + reward, steps + 1), \
               (obs, action, reward, terminated)
    
    # Run for up to 1000 steps or until the game ends
    final_carry, trajectory = jax.lax.scan(step_fn, (rng_play, state, observation, 0, 0), length=1000)
    
    return final_carry, trajectory

# Run it!
rng = jax.random.PRNGKey(42)
(final_rng, final_state, final_obs, total_reward, steps), trajectory = run_episode(rng)

print(f"Episode finished after {steps} steps with total reward: {total_reward}")
```

## Understanding What Just Happened

When you ran that code, several important things happened:

1. **JIT Compilation**: The first time you called `run_episode`, JAX compiled it into optimized code. This takes a moment, but subsequent calls are lightning fast.

2. **Functional Programming**: Notice how everything is immutable. Instead of modifying state in-place, we create new states with each step. This is how JAX achieves its performance and parallelization magic.

3. **Vectorization Ready**: While we ran a single episode, the same code can run thousands of episodes in parallel by changing the input shapes.

## Trying Different Games

Octax comes with many built-in games. Here are some favorites:

```python
# Classic Pong - simple and great for learning
pong_env, _ = create_environment("pong")

# Brix (Breakout) - more complex, great for RL training
brix_env, _ = create_environment("brix")

# Tetris - challenging and strategic
tetris_env, _ = create_environment("tetris")
```

Each game has different characteristics. Pong is simple with clear rewards, Brix requires spatial reasoning, and Tetris demands long-term planning.

## Visualizing Game Play

Want to see what's happening? Octax includes rendering capabilities:

```python
from octax.rendering import create_video

# Run an episode and collect states
@jax.jit
def collect_states(rng_key):
    def step_and_collect(carry, _):
        rng, state, obs = carry
        rng, rng_action = jax.random.split(rng)
        action = random_policy(rng_action, obs)
        next_state, next_obs, reward, terminated, truncated, info = env.step(state, action)
        return (rng, next_state, next_obs), next_state
    
    rng_reset, rng_play = jax.random.split(rng_key)
    state, obs, info = env.reset(rng_reset)
    
    final_carry, states = jax.lax.scan(step_and_collect, (rng_play, state, obs), length=500)
    return states

rng = jax.random.PRNGKey(123)
states = collect_states(rng)

# Create a video showing the gameplay
create_video(states, display=True)
```

This displays a video of your random agent playing the game.

## Using the Gymnasium Interface

If you're familiar with Gymnasium, Octax provides a compatible wrapper:

```python
from octax.gymnasium_wrapper import make_gymnasium_env

# Create a standard Gymnasium environment
gym_env = make_gymnasium_env("brix", render_mode="rgb_array")

print(f"Action space: {gym_env.action_space}")
print(f"Observation space: {gym_env.observation_space}")

# Standard Gymnasium loop
obs, info = gym_env.reset(seed=42)
for step in range(100):
    action = gym_env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = gym_env.step(action)
    
    if terminated or truncated:
        obs, info = gym_env.reset()

gym_env.close()
```

This interface is perfect if you want to use Octax with existing RL frameworks like Stable-Baselines3 or RLLib.

## Performance with JAX

JAX enables high-performance parallel execution. Here's an example running 1000 episodes:

```python
import time

# Time a batch of episodes
@jax.jit
def batch_episodes(rng_key, num_episodes=1000):
    """Run many episodes in parallel"""
    rngs = jax.random.split(rng_key, num_episodes)
    
    def single_episode(rng):
        state, obs, info = env.reset(rng)
        def step_fn(carry, _):
            rng, state, obs = carry
            rng, rng_action = jax.random.split(rng)
            action = random_policy(rng_action, obs)
            next_state, next_obs, reward, terminated, truncated, info = env.step(state, action)
            return (rng, next_state, next_obs), reward
        
        final_carry, rewards = jax.lax.scan(step_fn, (rng, state, obs), length=200)
        return jnp.sum(rewards)
    
    # Map over all episodes - this runs them in parallel!
    return jax.vmap(single_episode)(rngs)

rng = jax.random.PRNGKey(0)

# First run includes compilation time
start = time.time()
rewards = batch_episodes(rng, 1000)
compilation_time = time.time() - start

# Second run is pure execution
start = time.time()
rewards = batch_episodes(rng, 1000)
execution_time = time.time() - start

print(f"1000 episodes compiled in: {compilation_time:.2f}s")
print(f"1000 episodes executed in: {execution_time:.2f}s")
print(f"That's {1000/execution_time:.0f} episodes per second!")
```
