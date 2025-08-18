"""Stable Baselines3 JAX (SBX) PPO training example for Octax environments.

This example demonstrates how to use SBX PPO with Octax CHIP-8 game environments,
providing significant performance improvements over regular Stable Baselines3.
"""

import gymnasium as gym
from sbx import PPO
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Any, Optional
import os

from octax.gymnasium_wrapper import make_gymnasium_env


def train_sbx_ppo(
    game_name: str = "blinky",
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    verbose: int = 1,
    model_save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Train PPO agent using SBX on Octax environment.
    
    Args:
        game_name: Name of the Octax game environment
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for PPO
        n_steps: Number of steps per rollout
        batch_size: Batch size for training
        n_epochs: Number of epochs per update
        verbose: Verbosity level
        model_save_path: Path to save trained model (optional)
        
    Returns:
        Dictionary containing training results and metrics
    """
    print(f"Starting SBX PPO training on {game_name}")
    print(f"Training parameters:")
    print(f"  - Total timesteps: {total_timesteps:,}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Steps per rollout: {n_steps}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs per update: {n_epochs}")
    
    # Create environment using existing wrapper
    env = make_gymnasium_env(game_name)
    print(f"Created {game_name} environment")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space}")
    
    # Create PPO model with custom hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=verbose,
        device="auto",  # SBX will use JAX automatically
    )
    
    print(f"Created SBX PPO model with MlpPolicy")
    
    # Train the model
    print(f"Starting training...")
    start_time = time.time()
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Training completed!")
    print(f"Training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    print(f"Throughput: {total_timesteps/training_time:.0f} timesteps/second")
    
    # Save model if path provided
    if model_save_path:
        model.save(model_save_path)
        print(f"Model saved to: {model_save_path}")
    
    # Evaluate trained model
    print(f"Evaluating trained model...")
    eval_results = evaluate_model(model, env, n_episodes=10)
    
    results = {
        "model": model,
        "environment": env,
        "training_time": training_time,
        "throughput": total_timesteps / training_time,
        "eval_results": eval_results,
        "hyperparameters": {
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
        }
    }
    
    return results


def evaluate_model(model, env: gym.Env, n_episodes: int = 10) -> Dict[str, float]:
    """Evaluate trained model performance.
    
    Args:
        model: Trained SBX PPO model
        env: Environment to evaluate on
        n_episodes: Number of episodes for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if done or truncated:
                done = True
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_scores.append(info.get('score', 0))
        
        print(f"  Episode {episode+1}: Reward={episode_reward:.2f}, "
              f"Length={episode_length}, Score={info.get('score', 0)}")
    
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "mean_score": np.mean(episode_scores),
        "max_score": np.max(episode_scores),
        "all_rewards": episode_rewards,
        "all_scores": episode_scores,
    }
    
    print(f"Evaluation Results (n={n_episodes}):")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Episode Length: {results['mean_length']:.1f}")
    print(f"  Mean Score: {results['mean_score']:.2f}")
    print(f"  Max Score: {results['max_score']:.2f}")
    
    return results


def plot_evaluation_results(results: Dict[str, Any], save_path: str = "plots/sbx_evaluation.png"):
    """Plot evaluation results."""
    eval_data = results["eval_results"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot episode rewards
    episodes = range(1, len(eval_data["all_rewards"]) + 1)
    ax1.plot(episodes, eval_data["all_rewards"], 'b-o', linewidth=2, markersize=6)
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)
    
    # Add mean line
    mean_reward = eval_data["mean_reward"]
    ax1.axhline(y=mean_reward, color='r', linestyle='--', 
                label=f'Mean: {mean_reward:.2f}')
    ax1.legend()
    
    # Plot episode scores
    ax2.plot(episodes, eval_data["all_scores"], 'g-o', linewidth=2, markersize=6)
    ax2.set_title("Episode Scores")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Score")
    ax2.grid(True, alpha=0.3)
    
    # Add mean line
    mean_score = eval_data["mean_score"]
    ax2.axhline(y=mean_score, color='r', linestyle='--', 
                label=f'Mean: {mean_score:.2f}')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation plot saved to: {save_path}")
    
    # Try to show plot if display available
    try:
        plt.show()
    except Exception:
        print("Plot saved successfully (display not available)")
    
    plt.close()


def compare_multiple_games():
    """Train and compare PPO performance across multiple Octax games."""
    games = ["blinky", "pong", "brix", "tank"]
    results = {}
    
    print("Training PPO on multiple Octax games...")
    
    for game in games:
        print(f"\n{'='*50}")
        print(f"Training on {game.upper()}")
        print('='*50)
        
        try:
            game_results = train_sbx_ppo(
                game_name=game,
                total_timesteps=50000,  # Smaller for comparison
                verbose=1
            )
            results[game] = game_results
        except Exception as e:
            print(f"Error training on {game}: {e}")
            continue
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print('='*60)
    
    for game, result in results.items():
        eval_data = result["eval_results"]
        print(f"{game.upper():<15} | "
              f"Mean Reward: {eval_data['mean_reward']:>8.2f} | "
              f"Mean Score: {eval_data['mean_score']:>8.2f} | "
              f"Training Time: {result['training_time']:>6.1f}s")
    
    return results


def main():
    """Main example demonstrating SBX PPO with Octax."""
    print("Stable Baselines3 JAX (SBX) + Octax Example")
    print("=" * 50)
    
    # Single game training example
    results = train_sbx_ppo(
        game_name="blinky",
        total_timesteps=100000,
        learning_rate=3e-4,
        model_save_path="models/sbx_ppo_blinky"
    )
    
    # Plot results
    plot_evaluation_results(results)
    
    # Optional: Compare multiple games (uncomment to run)
    # print("\n" + "="*60)
    # compare_multiple_games()
    
    print(f"\nExample completed successfully!")
    return results


if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    main()