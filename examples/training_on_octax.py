"""Real-time PPO training demo with progress bars during JAX execution.

This demonstrates the scan_with_progress decorator that provides live updates
during JAX compilation and execution, not just after training completes.
"""

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

from octax.environments import create_environment
from octax.logging import scan_with_progress_and_metrics, ConsoleLogger


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
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(
            x
        )
        x = nn.tanh(x)
        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(
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
    """Create training function with real-time progress bars."""

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

        # Training loop with real-time progress bar AND live metrics
        @scan_with_progress_and_metrics(
            num_updates,
            desc=f"🚀 PPO Training ({config['TOTAL_TIMESTEPS']:,} timesteps)",
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
    print(f"📊 Training plots saved to: {plot_filename}")

    # Try to show plot if display is available
    try:
        plt.show()
    except Exception:
        print("📈 Plot saved successfully (display not available)")

    plt.close()


def main():
    """Real-time training demo with live progress bars."""
    logger = ConsoleLogger("RealTimeDemo")

    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 64,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e6,
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
    logger.info(
        f"Expected training time: ~{config['NUM_UPDATES'] * 2 / 60:.1f} minutes"
    )

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
        f"✅ Training completed in {elapsed_time/60:.1f} minutes ({elapsed_time:.1f}s)"
    )
    logger.info(
        f"⚡ Throughput: {config['TOTAL_TIMESTEPS']/elapsed_time:.0f} timesteps/second"
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
        logger.info("📈 Generating training plots...")
        plot_training_results(live_metrics, config)

    return out


if __name__ == "__main__":
    main()
