"""PPO training demo"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
from functools import partial
from typing import Tuple, Any
import time
import matplotlib.pyplot as plt
import cv2

from octax.environments import create_environment
from octax.logging import scan_with_progress_and_metrics, ConsoleLogger
from octax.gymnasium_wrapper import make_gymnasium_env


def categorical_sample(rng, logits):
    """Sample from categorical distribution."""
    return jax.random.categorical(rng, logits)


def categorical_log_prob(logits, action):
    """Calculate log probability of action under categorical distribution."""
    log_probs = jax.nn.log_softmax(logits)
    return log_probs[action]


def categorical_entropy(logits):
    """Calculate entropy of categorical distribution."""
    probs = jax.nn.softmax(logits)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(probs * log_probs, axis=-1)


class ActorCritic(nn.Module):
    """Simple Actor-Critic network for PPO."""

    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Flatten observation
        x = x.reshape((x.shape[0], -1))

        # Shared layers
        x = nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(
            x
        )
        x = nn.tanh(x)
        x = nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(
            x
        )
        x = nn.tanh(x)
        x = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(
            x
        )
        x = nn.tanh(x)

        # Actor head (policy)
        actor = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)

        # Critic head (value function)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)

        return actor, jnp.squeeze(critic, axis=-1)


def make_train_with_progress(config):

    # Create environment
    env, metadata = create_environment("blinky")
    num_updates = config["NUM_UPDATES"]

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # Initialize network
        network = ActorCritic(env.num_actions)

        rng, _rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((1, config["FRAME_SKIP"], 64, 32))
        init_params = network.init(_rng, dummy_obs)

        # Initialize optimizer
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(0.5),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(0.5),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=init_params,
            tx=tx,
        )

        # Initialize environment states
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        env_state, obsv, info = jax.vmap(env.reset)(reset_rng)

        @scan_with_progress_and_metrics(
            num_updates,
            desc=f"PPO Training ({config['TOTAL_TIMESTEPS']:,} timesteps)",
            metric_keys=["score", "loss", "reward", "entropy"],  # Show key metrics
            print_rate=max(1, num_updates // 20),  # Update every 5%
            colour="green",
            leave=True,
        )
        def _update_step(runner_state, step):
            train_state, env_state, last_obs, rng = runner_state

            # Collect rollout
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # Get action and value
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action_rng = jax.random.split(_rng, config["NUM_ENVS"])
                action = jax.vmap(categorical_sample, in_axes=(0, 0))(action_rng, pi)
                log_prob = jax.vmap(categorical_log_prob, in_axes=(0, 0))(pi, action)

                # Step environment
                env_state, obsv, reward, done, truncated, info = jax.vmap(env.step)(
                    env_state, action
                )

                transition = {
                    "done": done,
                    "action": action,
                    "value": value,
                    "reward": reward,
                    "log_prob": log_prob,
                    "obs": last_obs,
                    "info": info,
                }
                return (train_state, env_state, obsv, rng), transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # Calculate advantages and returns
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition["done"],
                        transition["value"],
                        transition["reward"],
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch["value"]

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # Update network
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # Get current policy and value
                        pi, value = network.apply(params, traj_batch["obs"])
                        log_prob = jax.vmap(categorical_log_prob, in_axes=(0, 0))(
                            pi, traj_batch["action"]
                        )

                        # Policy loss
                        ratio = jnp.exp(log_prob - traj_batch["log_prob"])
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()

                        # Value loss
                        value_pred_clipped = traj_batch["value"] + (
                            value - traj_batch["value"]
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # Entropy loss
                        entropy = jax.vmap(categorical_entropy)(pi).mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state

                # Shuffle batch
                rng, _rng = jax.random.split(rng)
                batch_size = config["NUM_ENVS"] * config["NUM_STEPS"]
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )

                # Update minibatches
                minibatch_size = batch_size // config["NUM_MINIBATCHES"]
                minibatches = jax.tree.map(
                    lambda x: x.reshape(
                        (config["NUM_MINIBATCHES"], minibatch_size) + x.shape[1:]
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # Extract metrics for both live display and full tracking
            current_score = jnp.mean(
                traj_batch["info"]["score"][-1]
            )  # Final score averaged over envs
            current_reward = jnp.mean(
                jnp.sum(traj_batch["reward"], axis=0)
            )  # Sum over time, mean over envs

            # Get final loss and entropy from the last loss_info
            # loss_info structure: (total_loss, (value_loss, policy_loss, entropy))
            final_loss = loss_info[0][
                -1
            ]  # total_loss from last minibatch of last epoch
            aux_info = loss_info[1][
                -1
            ]  # auxiliary info from last minibatch of last epoch
            final_entropy = aux_info[2]  # entropy is third element in aux info

            # Live metrics for real-time display
            live_metrics = {
                "score": current_score,
                "loss": final_loss,
                "reward": current_reward,
                "entropy": final_entropy,
            }

            # Full metrics for final analysis
            full_metrics = traj_batch["info"]

            runner_state = (train_state, env_state, last_obs, rng)

            # Return in format: (new_carry, (live_metrics, full_metrics))
            return runner_state, (live_metrics, full_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)

        # Run training loop with progress bar and live metrics
        runner_state, outputs = jax.lax.scan(
            _update_step, runner_state, jnp.arange(num_updates)
        )

        # Separate live metrics from full metrics
        live_metrics, full_metrics = outputs

        return {
            "runner_state": runner_state,
            "live_metrics": live_metrics,
            "metrics": full_metrics,
        }

    return train


def evaluate_trained_policy(
    train_state, network, config, rng, game_name="blinky", n_episodes=64, max_steps=2000
):
    env, metadata = create_environment(game_name)

    @partial(jax.jit, static_argnums=(0,))
    def eval_episodes(network_apply, params, rng):
        # Initialize vectorized environments
        reset_rng = jax.random.split(rng, n_episodes)
        env_state, obsv, info = jax.vmap(env.reset)(reset_rng)

        def eval_step(carry, unused):
            env_state, obs = carry

            # Get deterministic actions from policy
            pi, value = network_apply(params, obs)
            action = jnp.argmax(pi, axis=-1)  # Deterministic policy

            # Step environments
            env_state, obs, reward, done, truncated, info = jax.vmap(env.step)(
                env_state, action
            )

            return (env_state, obs), {
                "reward": reward,
                "done": done,
                "truncated": truncated,
                "info": info,
            }

        # Run evaluation episodes
        initial_carry = (env_state, obsv)
        final_carry, eval_data = jax.lax.scan(
            eval_step, initial_carry, None, length=max_steps
        )

        return eval_data

    # Run JIT-compiled evaluation
    eval_data = eval_episodes(network.apply, train_state.params, rng)

    # Extract metrics from evaluation data
    episode_rewards = jnp.sum(eval_data["reward"], axis=0)  # Sum over time steps
    episode_scores = eval_data["info"]["score"][-1]  # Final scores
    episode_lengths = jnp.sum(1 - eval_data["done"], axis=0)  # Count non-done steps

    results = {
        "mean_reward": jnp.mean(episode_rewards),
        "std_reward": jnp.std(episode_rewards),
        "mean_length": jnp.mean(episode_lengths),
        "mean_score": jnp.mean(episode_scores),
        "max_score": jnp.max(episode_scores),
        "min_score": jnp.min(episode_scores),
        "all_rewards": episode_rewards,
        "all_scores": episode_scores,
    }

    return results


def demonstrate_policy(train_state, network, rng, game_name="blinky", n_steps=2000):
    env, metadata = create_environment(game_name)

    @partial(jax.jit, static_argnums=(0,))
    def demo_episode(network_apply, params, rng):
        # Reset environment for demonstration
        env_state, obs, info = env.reset(rng)

        def demo_step(carry, unused):
            env_state, obs = carry

            # Get action from policy (deterministic)
            pi, value = network_apply(params, obs[None])  # Add batch dim
            action = jnp.argmax(pi[0])  # Take most likely action

            # Step environment
            env_state, obs, reward, done, truncated, info = env.step(env_state, action)

            return (env_state, obs), {
                "action": action,
                "reward": reward,
                "done": done,
                "truncated": truncated,
                "info": info,
                "value": value,
            }

        # Run demonstration episode
        initial_carry = (env_state, obs)
        final_carry, demo_data = jax.lax.scan(
            demo_step, initial_carry, None, length=n_steps
        )

        return demo_data

    # Run JIT-compiled demonstration
    demo_data = demo_episode(network.apply, train_state.params, rng)

    # Extract final metrics
    total_reward = jnp.sum(demo_data["reward"])
    final_score = demo_data["info"]["score"][-1]  # Last score
    episode_length = jnp.sum(1 - demo_data["done"])  # Count non-done steps

    # Action distribution
    actions = demo_data["action"]
    action_counts = jnp.array([jnp.sum(actions == i) for i in range(env.num_actions)])

    return {
        "final_score": final_score,
        "total_reward": total_reward,
        "episode_length": episode_length,
        "action_counts": action_counts,
        "actions": actions,
        "rewards": demo_data["reward"],
        "values": demo_data["value"],
        "scores": demo_data["info"]["score"],
    }


def demonstrate_policy_video(train_state, network, game_name="blinky", max_steps=1000):
    """Demonstrate trained policy with video playback like in gymnasium_example.py."""
    print(f"\nDemonstrating trained policy on {game_name.upper()} with video...")

    # Create Gymnasium environment with rendering (like the example)
    env = make_gymnasium_env(
        game_name, render_mode="rgb_array", render_scale=8, color_scheme="octax"
    )

    def trained_policy(observation):
        """Use the trained policy to select actions."""
        obs_jax = jnp.array(observation)
        pi, value = network.apply(train_state.params, obs_jax[None])  # Add batch dim
        action = int(jnp.argmax(pi[0]))  # Deterministic policy
        return action

    # Start episode
    obs, info = env.reset(seed=42)
    total_reward = 0
    episode_count = 0
    frames = []

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Starting policy demonstration...")

    start_time = time.time()

    for step in range(max_steps):
        # Use trained policy instead of random
        action = trained_policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Collect frames from the beginning (not after 4 episodes like the example)
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        # Print progress
        if step % 100 == 0:
            print(
                f"  Step {step}: Score={info.get('score', 0)}, Reward={total_reward:.2f}"
            )

        if terminated or truncated:
            episode_count += 1
            print(
                f"  Episode {episode_count} ended. Total reward: {total_reward:.2f}, Score: {info.get('score', 0)}"
            )
            obs, info = env.reset()
            total_reward = 0

            # Stop after a few episodes for demonstration
            if episode_count >= 3:
                break

    end_time = time.time()
    print(f"Steps per second: {len(frames) / (end_time - start_time):.1f}")
    print(f"Total episodes: {episode_count}")

    # Save video and display frames
    if frames:
        print(f"Captured {len(frames)} frames")

        # Save video file
        save_video(frames, f"videos/{game_name}_policy_demo.mp4", fps=10)

        # Display frames (like the gymnasium example)
        print(f"Displaying video (press 'q' to quit)")
        for i, frame in enumerate(frames):
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Add frame counter overlay
            cv2.putText(
                frame_bgr,
                f"Frame {i+1}/{len(frames)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(f"{game_name.upper()} Policy Demo", frame_bgr)

            # Control playback speed and allow quit
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        print(f"Video demonstration completed!")

    env.close()

    return {
        "episodes": episode_count,
        "frames_captured": len(frames),
        "video_file": f"videos/{game_name}_policy_demo.mp4",
    }


def save_video(frames, filename, fps=10):
    """Save frames as a video file."""
    import os

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if not frames:
        print("No frames to save")
        return

    # Get frame dimensions
    height, width, channels = frames[0].shape

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video saved to: {filename}")


def print_evaluation_results(eval_results, demo_results):
    """Print evaluation and demonstration results."""
    print(f"\nEvaluation Results:")
    print(
        f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}"
    )
    print(f"  Mean Episode Length: {eval_results['mean_length']:.1f}")
    print(f"  Mean Score: {eval_results['mean_score']:.2f}")
    print(f"  Max Score: {eval_results['max_score']:.2f}")
    print(f"  Min Score: {eval_results['min_score']:.2f}")

    print(f"\nDemonstration Episode:")
    print(f"  Final Score: {demo_results['final_score']:.0f}")
    print(f"  Total Reward: {demo_results['total_reward']:.2f}")
    print(f"  Episode Length: {demo_results['episode_length']:.0f} steps")

    # Action distribution
    action_counts = demo_results["action_counts"]
    total_actions = jnp.sum(action_counts)
    print(f"\nAction Distribution:")
    for i, count in enumerate(action_counts):
        percentage = (count / total_actions) * 100 if total_actions > 0 else 0
        print(f"  Action {i}: {count:4.0f} times ({percentage:5.1f}%)")


def plot_training_results(live_metrics, config):
    """Plot training metrics from PPO training."""
    # Extract metrics arrays
    scores = live_metrics["score"]
    losses = live_metrics["loss"]
    rewards = live_metrics["reward"]
    entropies = live_metrics["entropy"]

    # Create update steps for x-axis
    update_steps = jnp.arange(len(scores))
    timesteps = update_steps * config["NUM_ENVS"] * config["NUM_STEPS"]

    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f'PPO Training Results ({config["TOTAL_TIMESTEPS"]:,} timesteps)',
        fontsize=14,
        fontweight="bold",
    )

    # Plot 1: Score over time
    ax1.plot(timesteps, scores, "b-", linewidth=1.5, alpha=0.8)
    ax1.set_title("Game Score")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Score")
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

    # Plot 2: Loss over time (handle multidimensional arrays)
    loss_values = jnp.mean(losses, axis=-1) if losses.ndim > 1 else losses
    ax2.plot(timesteps, loss_values, "r-", linewidth=1.5, alpha=0.8)
    ax2.set_title("Training Loss")
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

    # Plot 3: Episode rewards
    ax3.plot(timesteps, rewards, "g-", linewidth=1.5, alpha=0.8)
    ax3.set_title("Episode Reward")
    ax3.set_xlabel("Timesteps")
    ax3.set_ylabel("Reward")
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

    # Plot 4: Policy entropy (handle multidimensional arrays)
    entropy_values = jnp.mean(entropies, axis=-1) if entropies.ndim > 1 else entropies
    ax4.plot(timesteps, entropy_values, "orange", linewidth=1.5, alpha=0.8)
    ax4.set_title("Policy Entropy")
    ax4.set_xlabel("Timesteps")
    ax4.set_ylabel("Entropy")
    ax4.grid(True, alpha=0.3)
    ax4.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

    # Add final values as text annotations
    final_score = float(scores[-1])
    final_loss = float(jnp.mean(losses[-1]))  # Handle multidimensional loss
    final_reward = float(rewards[-1])
    final_entropy = float(jnp.mean(entropies[-1]))  # Handle multidimensional entropy

    ax1.text(
        0.02,
        0.98,
        f"Final: {final_score:.1f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    ax2.text(
        0.02,
        0.98,
        f"Final: {final_loss:.4f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    ax3.text(
        0.02,
        0.98,
        f"Final: {final_reward:.2f}",
        transform=ax3.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    ax4.text(
        0.02,
        0.98,
        f"Final: {final_entropy:.3f}",
        transform=ax4.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    # Save plot to file and optionally display
    import os

    os.makedirs("plots", exist_ok=True)
    plot_filename = "plots/ppo_training_results.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Training plots saved to: {plot_filename}")

    # Try to show plot if display is available
    try:
        plt.show()
    except Exception:
        print("Plot saved successfully (display not available)")

    plt.close()


def main():
    logger = ConsoleLogger("Demo")

    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 64,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "ANNEAL_LR": True,
        "FRAME_SKIP": 4,
    }

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    logger.info(
        f"Training for {config['TOTAL_TIMESTEPS']:,} timesteps ({config['NUM_UPDATES']} updates)"
    )
    logger.info(f"Using {config['NUM_ENVS']} vectorized environments")

    # Create and JIT compile training function
    rng = jax.random.PRNGKey(42)
    train_fn = make_train_with_progress(config)
    train_jit = jax.jit(train_fn)

    start_time = time.time()
    out = train_jit(rng)
    end_time = time.time()

    # Training completed - show comprehensive results
    elapsed_time = end_time - start_time
    logger.info(
        f"Training completed in {elapsed_time/60:.1f} minutes ({elapsed_time:.1f}s)"
    )
    logger.info(
        f"Throughput: {config['TOTAL_TIMESTEPS']/elapsed_time:.0f} timesteps/second"
    )

    # Analyze live metrics
    if "live_metrics" in out:
        live_metrics = out["live_metrics"]
        final_score = float(jnp.mean(live_metrics["score"][-1]))
        final_loss = float(jnp.mean(live_metrics["loss"][-1]))
        final_reward = float(jnp.mean(live_metrics["reward"][-1]))
        final_entropy = float(jnp.mean(live_metrics["entropy"][-1]))

        logger.info(f"Final Results:")
        logger.info(f"  Score: {final_score:.2f}")
        logger.info(f"  Loss: {final_loss:.4f}")
        logger.info(f"  Avg Reward: {final_reward:.3f}")
        logger.info(f"  Entropy: {final_entropy:.3f}")

        # Plot training results
        logger.info("Generating training plots...")
        plot_training_results(live_metrics, config)

        # Evaluate and demonstrate trained policy
        game_name = "blinky"
        logger.info("Evaluating trained policy...")
        train_state = out["runner_state"][0]  # Extract train_state from runner_state
        env_for_eval, _ = create_environment(
            game_name
        )  # Create environment for evaluation
        network = ActorCritic(
            env_for_eval.num_actions
        )  # Recreate network for evaluation

        # Create RNG keys for evaluation
        rng, eval_rng = jax.random.split(jax.random.PRNGKey(42))
        rng, demo_rng = jax.random.split(rng)

        # Parallel evaluation (64 episodes)
        eval_results = evaluate_trained_policy(
            train_state, network, config, eval_rng, game_name, n_episodes=64
        )
        demo_results = demonstrate_policy(
            train_state, network, demo_rng, game_name, n_steps=2000
        )

        # Print evaluation results
        print_evaluation_results(eval_results, demo_results)

        # Video policy demonstration with OpenCV
        logger.info("Running video policy demonstration...")
        video_results = demonstrate_policy_video(
            train_state, network, game_name, max_steps=500
        )

    return out


if __name__ == "__main__":
    main()
