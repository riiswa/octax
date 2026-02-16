# Reinforcement Learning Environments

Octax transforms classic CHIP-8 games into modern reinforcement learning environments. This tutorial explores how to use Octax for training RL agents, understanding reward structures, and leveraging JAX's performance advantages for large-scale experiments.

## From Games to RL Environments

When you play a CHIP-8 game manually, you see pixels on screen and press buttons based on what you observe. An RL environment formalizes this interaction into the standard observation-action-reward cycle. Octax wraps the core emulator with additional tracking for scores, termination conditions, and standardized interfaces.

Let's start by understanding how a game becomes an RL environment:

```python
import jax
import jax.numpy as jnp
from octax.environments import create_environment

# Create the Brix environment (Breakout clone)
env, metadata = create_environment("brix")

print(f"Game: {metadata['title']}")
print(f"Actions available: {env.num_actions}")
print(f"Action meanings: {env.action_names}")
print(f"Observation shape: {env.observation_space}")
```

Octax automatically detects important game elements:
- **Actions**: Which CHIP-8 keys are actually used (not all 16 keys are needed)
- **Score**: Which memory register tracks the player's score
- **Termination**: When the game ends (lives = 0, game over screen, etc.)
- **Observations**: The 64x32 pixel display as the agent's "vision"

## The Environment Interface

Octax environments follow a standard interface similar to Gymnasium but optimized for JAX:

```python
def explore_environment_interface():
    env, _ = create_environment("pong")
    
    # Reset gives you initial state and observation
    rng = jax.random.PRNGKey(42)
    state, observation, info = env.reset(rng)
    
    print(f"Initial observation shape: {observation.shape}")
    print(f"Observation is binary: {jnp.all((observation == 0) | (observation == 1))}")
    print(f"Info keys: {info.keys() if isinstance(info, dict) else 'No info dict'}")
    
    # Step takes action and returns new state
    action = 1  # Press right
    new_state, new_obs, reward, terminated, truncated, info = env.step(state, action)
    
    print(f"After one step:")
    print(f"  Reward: {reward}")
    print(f"  Terminated: {terminated}")
    print(f"  Truncated: {truncated}")
    
    return new_state, new_obs

state, obs = explore_environment_interface()
```

The observation is always the raw CHIP-8 display—a 64x32 binary array. This gives agents complete visual information about the game state, just like a human player would see.

## Understanding Rewards and Scoring

Different games have different reward structures. Let's examine how Octax detects and uses scores:

```python
def analyze_game_scoring():
    """Compare scoring mechanisms across different games"""
    games = ["pong", "brix", "tetris", "deep"]
    
    for game_name in games:
        env, metadata = create_environment(game_name)
        
        # Reset and run a few random steps
        rng = jax.random.PRNGKey(123)
        state, obs, info = env.reset(rng)
        
        total_reward = 0
        for step in range(100):
            rng, rng_action = jax.random.split(rng)
            action = jax.random.randint(rng_action, (), 0, env.num_actions)
            state, obs, reward, terminated, truncated, info = env.step(state, action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"{game_name.upper()}:")
        print(f"  Total reward after {step+1} steps: {total_reward}")
        print(f"  Final score: {env.score_fn(state)}")
        print(f"  Game ended: {terminated or truncated}")
        print()

analyze_game_scoring()
```

Most Octax environments use score-based rewards—the reward at each step is the change in score since the last step. This encourages agents to maximize their game score, which aligns with human objectives.

## Training Your First RL Agent

Let's implement a simple policy gradient agent using JAX. This demonstrates how Octax's functional design makes RL training natural:

```python
import jax
import jax.numpy as jnp
from functools import partial

def create_simple_policy(observation_shape, num_actions):
    """Create a simple neural network policy"""
    def init_params(rng):
        # Simple feedforward network: flatten -> hidden -> actions
        input_size = observation_shape[0] * observation_shape[1]
        
        k1, k2 = jax.random.split(rng)
        return {
            'w1': jax.random.normal(k1, (input_size, 128)) * 0.1,
            'b1': jnp.zeros(128),
            'w2': jax.random.normal(k2, (128, num_actions)) * 0.1,
            'b2': jnp.zeros(num_actions)
        }
    
    def forward(params, observation):
        x = observation.reshape(-1)  # Flatten
        x = jnp.tanh(x @ params['w1'] + params['b1'])  # Hidden layer
        logits = x @ params['w2'] + params['b2']  # Output layer
        return logits
    
    def sample_action(params, observation, rng):
        logits = forward(params, observation)
        return jax.random.categorical(rng, logits)
    
    return init_params, forward, sample_action

# Create the policy
env, _ = create_environment("pong")
init_params, forward, sample_action = create_simple_policy(
    (64, 32), env.num_actions
)

# Initialize policy parameters
rng = jax.random.PRNGKey(0)
params = init_params(rng)

print(f"Policy parameters:")
for key, value in params.items():
    print(f"  {key}: shape {value.shape}")
```

Now let's train this policy using a simple REINFORCE algorithm:

```python
@jax.jit
def collect_trajectory(params, env, rng, episode_length=1000):
    """Collect a full trajectory using current policy"""
    
    def step_fn(carry, _):
        rng, state, obs, total_reward = carry
        rng, rng_action, rng_reset = jax.random.split(rng, 3)
        
        # Sample action from policy
        action = sample_action(params, obs, rng_action)
        
        # Take environment step
        next_state, next_obs, reward, terminated, truncated, info = env.step(state, action)
        
        # Reset if episode ended
        reset_needed = terminated | truncated
        next_state, next_obs, info = jax.lax.cond(
            reset_needed,
            lambda _: env.reset(rng_reset),
            lambda _: (next_state, next_obs, info),
            None
        )
        
        new_carry = (rng, next_state, next_obs, total_reward + reward)
        step_data = (obs, action, reward, reset_needed)
        
        return new_carry, step_data
    
    # Initialize episode
    rng, rng_reset = jax.random.split(rng)
    state, obs, info = env.reset(rng_reset)
    
    # Collect trajectory
    final_carry, trajectory = jax.lax.scan(
        step_fn, (rng, state, obs, 0.0), length=episode_length
    )
    
    observations, actions, rewards, dones = trajectory
    total_reward = final_carry[3]
    
    return observations, actions, rewards, dones, total_reward

def compute_returns(rewards, dones, gamma=0.99):
    """Compute discounted returns for REINFORCE"""
    def scan_fn(carry, inputs):
        reward, done = inputs
        discounted_return = carry
        new_return = reward + gamma * discounted_return * (1 - done)
        return new_return, new_return
    
    # Reverse to compute returns backwards
    _, returns = jax.lax.scan(
        scan_fn, 0.0, (rewards[::-1], dones[::-1])
    )
    return returns[::-1]

@jax.jit
def policy_gradient_update(params, trajectory_data, learning_rate=1e-4):
    """Update policy using REINFORCE gradient"""
    observations, actions, rewards, dones = trajectory_data
    returns = compute_returns(rewards, dones)
    
    # Normalize returns
    returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-8)
    
    def loss_fn(params):
        logits = jax.vmap(forward, in_axes=(None, 0))(params, observations)
        log_probs = jax.nn.log_softmax(logits)
        
        # Select log probabilities for taken actions
        action_log_probs = log_probs[jnp.arange(len(actions)), actions]
        
        # Policy gradient loss
        loss = -jnp.mean(action_log_probs * returns)
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    
    # Simple SGD update
    new_params = jax.tree_map(
        lambda p, g: p - learning_rate * g, params, grads
    )
    
    return new_params, loss

# Training loop
rng = jax.random.PRNGKey(42)
params = init_params(rng)

for episode in range(10):
    rng, rng_traj = jax.random.split(rng)
    
    # Collect trajectory
    obs, actions, rewards, dones, total_reward = collect_trajectory(
        params, env, rng_traj, episode_length=500
    )
    
    # Update policy
    params, loss = policy_gradient_update(params, (obs, actions, rewards, dones))
    
    print(f"Episode {episode}: Total reward = {total_reward:.1f}, Loss = {loss:.4f}")

print("Training complete! The agent learned to play through policy gradients.")
```

## Vectorized Training

Octax's real power comes from vectorization—training many agents in parallel. Here's how to run multiple environments simultaneously:

```python
@partial(jax.jit, static_argnums=1)
def vectorized_rollout(rng, num_envs=64):
    """Run multiple environments in parallel"""
    env, _ = create_environment("brix")
    
    # Split random keys for each environment
    rngs = jax.random.split(rng, num_envs)
    
    def single_env_rollout(rng):
        """Rollout for a single environment"""
        def step_fn(carry, _):
            rng, state, obs, total_reward = carry
            rng, rng_action = jax.random.split(rng)
            
            # Random policy for this demo
            action = jax.random.randint(rng_action, (), 0, env.num_actions)
            next_state, next_obs, reward, terminated, truncated, info = env.step(state, action)
            
            # Simple reset if needed
            new_carry = jax.lax.cond(
                terminated | truncated,
                lambda _: (rng, *env.reset(rng), total_reward + reward),
                lambda _: (rng, next_state, next_obs, total_reward + reward),
                None
            )
            
            return new_carry, reward
        
        # Initialize environment
        rng, rng_reset = jax.random.split(rng)
        state, obs, info = env.reset(rng_reset)
        
        # Run episode
        final_carry, rewards = jax.lax.scan(
            step_fn, (rng, state, obs, 0.0), length=1000
        )
        
        return final_carry[3]  # Return total reward
    
    # Map over all environments
    total_rewards = jax.vmap(single_env_rollout)(rngs)
    return total_rewards

# Run 64 environments in parallel
import time

rng = jax.random.PRNGKey(0)

# First run includes compilation
start = time.time()
rewards = vectorized_rollout(rng, 64)
compile_time = time.time() - start

# Second run is pure execution
start = time.time()
rewards = vectorized_rollout(rng, 64)
execution_time = time.time() - start

print(f"64 parallel episodes:")
print(f"  Compilation time: {compile_time:.2f}s")
print(f"  Execution time: {execution_time:.2f}s")
print(f"  Episodes per second: {64/execution_time:.0f}")
print(f"  Mean reward: {jnp.mean(rewards):.2f}")
print(f"  Reward std: {jnp.std(rewards):.2f}")
```

This vectorization capability enables efficient scaling from single-environment experiments to large parallel training runs.

## Environment Customization

Each game environment can be customized for specific research needs:

```python
def explore_environment_customization():
    """Demonstrate environment customization options"""
    
    # Different games have different characteristics
    games_info = []
    
    for game in ["pong", "brix", "tetris", "missile", "deep"]:
        env, metadata = create_environment(game)
        
        # Test episode to get characteristics
        rng = jax.random.PRNGKey(42)
        state, obs, info = env.reset(rng)
        
        episode_length = 0
        total_reward = 0
        
        for _ in range(1000):
            rng, rng_action = jax.random.split(rng)
            action = jax.random.randint(rng_action, (), 0, env.num_actions)
            state, obs, reward, terminated, truncated, info = env.step(state, action)
            
            episode_length += 1
            total_reward += reward
            
            if terminated or truncated:
                break
        
        games_info.append({
            'name': game,
            'actions': env.num_actions,
            'episode_length': episode_length,
            'final_reward': total_reward,
            'description': metadata.get('description', '')[:100] + '...'
        })
    
    # Display comparison
    print("Game Environment Comparison:")
    print("-" * 80)
    for info in games_info:
        print(f"{info['name'].upper():<12} | Actions: {info['actions']} | "
              f"Episode: {info['episode_length']:4d} steps | "
              f"Reward: {info['final_reward']:6.1f}")
    
    return games_info

game_comparison = explore_environment_customization()
```

## Integration with Popular RL Libraries

Octax includes a Gymnasium wrapper for compatibility with existing RL frameworks:

```python
from octax.gymnasium_wrapper import make_gymnasium_env
import time

# Create standard Gymnasium environment
gym_env = make_gymnasium_env("brix", render_mode="rgb_array")

print(f"Gymnasium Environment:")
print(f"  Action space: {gym_env.action_space}")
print(f"  Observation space: {gym_env.observation_space}")
print(f"  Spec: {gym_env.spec}")

# Standard Gymnasium interface
obs, info = gym_env.reset(seed=42)
done = False
step_count = 0
total_reward = 0

start_time = time.time()

while not done and step_count < 1000:
    action = gym_env.action_space.sample()
    obs, reward, terminated, truncated, info = gym_env.step(action)
    
    total_reward += reward
    step_count += 1
    done = terminated or truncated

end_time = time.time()

print(f"\nGaming Session Results:")
print(f"  Steps: {step_count}")
print(f"  Total reward: {total_reward}")
print(f"  FPS: {step_count/(end_time-start_time):.1f}")
print(f"  Episode ended: {done}")

gym_env.close()
```

This wrapper makes Octax environments compatible with libraries like Stable-Baselines3, RLLib, and others that expect the standard Gymnasium interface.
