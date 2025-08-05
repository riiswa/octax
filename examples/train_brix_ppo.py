"""Simple PPO training example for Brix using JAX vectorization.

This example demonstrates how to train a PPO agent on the Brix (Breakout) game
using JAX's vectorization capabilities for high-performance training.
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

from octax.environments import create_environment


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
        x = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(
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


def make_train(config):
    """Create training function with given config."""

    # Create environment
    env, metadata = create_environment("brix")

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

        # Training loop
        def _update_step(runner_state, unused):
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

            # Metrics
            metric = traj_batch["info"]
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)

        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metric}

    return train


def main():
    """Train PPO agent on Brix."""
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 64,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "ANNEAL_LR": True,
        "FRAME_SKIP": 4,
    }

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    print(f"Training PPO on Brix for {config['TOTAL_TIMESTEPS']:,} timesteps")
    print(f"Using {config['NUM_ENVS']} vectorized environments")
    print(f"Total updates: {config['NUM_UPDATES']}")

    # Create and JIT compile training function
    rng = jax.random.PRNGKey(42)
    train_jit = jax.jit(make_train(config))

    # Train
    start_time = time.time()
    out = train_jit(rng)
    end_time = time.time()

    # Results
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")

    # Print final metrics
    final_metrics = jax.tree.map(lambda x: x[-1], out["metrics"])
    mean_score = jnp.mean(final_metrics["score"])
    print(f"Final mean score: {mean_score:.2f}")

    return out


if __name__ == "__main__":
    main()
