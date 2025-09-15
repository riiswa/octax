import chex
import jax
import optax
from flax import struct
from flax.training.train_state import TrainState
import jax.numpy as jnp
from rejax import Algorithm
from rejax.algos.algorithm import register_init
from rejax.algos.mixins import OnPolicyMixin

from flax import linen as nn
from rejax.algos.ppo import AdvantageMinibatch, Trajectory
from rejax.networks import DiscretePolicy, VNetwork

class Agent(nn.Module):
    action_dim: int

    def setup(self) -> None:
        self.features = nn.Sequential([
                nn.Conv(features=32, kernel_size=(8, 4), strides=(4, 2)),
                nn.relu,
                nn.Conv(features=64, kernel_size=4, strides=2),
                nn.relu,
                nn.Conv(features=64, kernel_size=3, strides=1),
                nn.relu,
                lambda x: x.reshape(x.shape[0], -1)
            ]
        )

        self.actor = DiscretePolicy(
            action_dim=self.action_dim,
            hidden_layer_sizes=[256],
            activation=nn.relu,
        )

        self.critic = VNetwork(hidden_layer_sizes=[256],activation=nn.relu)

    def __call__(self, obs, rng, action=None):
        features = self.features(obs)

        value = self.critic(features)

        #  return action, log_prob, entropy, value
        if action is None:
            return *self.actor(features, rng), value
        else:
            return action, *self.actor.log_prob_entropy(features, action), value

    def call_critic(self, obs):
        features = self.features(obs)
        return self.critic(features)

    def call_actor(self, obs, rng):
        features = self.features(obs)
        return self.actor(features, rng)


class PPOOctax(OnPolicyMixin, Algorithm):
    agent: nn.Module = struct.field(pytree_node=False, default=None)
    num_epochs: int = struct.field(pytree_node=False, default=8)
    gae_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.95)
    clip_eps: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    vf_coef: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    ent_coef: chex.Scalar = struct.field(pytree_node=True, default=0.01)

    def make_act(self, ts):
        def act(obs, rng):
            obs = jnp.expand_dims(obs, 0)
            action, _, _, _ = self.agent.apply(ts.agent_ts.params, obs, rng)
            return jnp.squeeze(action)

        return act


    @classmethod
    def create_agent(cls, config, env, env_params):
        return Agent(action_dim=env.action_space(env_params).n)

    @register_init
    def initialize_network_params(self, rng):
        obs_ph = jnp.empty([1, *self.env.observation_space(self.env_params).shape])
        agent_params = self.agent.init(rng, obs_ph, rng)

        tx = optax.chain(
            optax.clip(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )

        return {"agent_ts": TrainState.create(apply_fn=(), params=agent_params, tx=tx)}

    def train_iteration(self, ts):
        ts, trajectories = self.collect_trajectories(ts)

        last_val = self.agent.apply(
            ts.agent_ts.params, ts.last_obs, method="call_critic"
        )

        last_val = jnp.where(ts.last_done, 0, last_val)
        advantages, targets = self.calculate_gae(trajectories, last_val)

        def update_epoch(ts, unused):
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            batch = AdvantageMinibatch(trajectories, advantages, targets)
            minibatches = self.shuffle_and_split(batch, minibatch_rng)
            ts, _ = jax.lax.scan(
                lambda ts, mbs: (self.update(ts, mbs), None),
                ts,
                minibatches,
            )
            return ts, None

        ts, _ = jax.lax.scan(update_epoch, ts, None, self.num_epochs)
        return ts

    def collect_trajectories(self, ts):
        def env_step(ts, unused):
            # Get keys for sampling action and stepping environment
            rng, new_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            rng_steps, rng_action = jax.random.split(new_rng, 2)
            rng_steps = jax.random.split(rng_steps, self.num_envs)


            action, log_prob, _, value = self.agent.apply(ts.agent_ts.params, ts.last_obs, rng_action)

            # Step environment
            t = self.vmap_step(rng_steps, ts.env_state, action, self.env_params)
            next_obs, env_state, reward, done, _ = t

            # Return updated runner state and transition
            transition = Trajectory(
                ts.last_obs, action, log_prob, reward, value, done
            )
            ts = ts.replace(
                env_state=env_state,
                last_obs=next_obs,
                last_done=done,
                global_step=ts.global_step + self.num_envs,
            )
            return ts, transition

        ts, trajectories = jax.lax.scan(env_step, ts, None, self.num_steps)
        return ts, trajectories

    def calculate_gae(self, trajectories, last_val):
        def get_advantages(advantage_and_next_value, transition):
            advantage, next_value = advantage_and_next_value
            delta = (
                    transition.reward.squeeze()  # For gymnax envs that return shape (1, )
                    + self.gamma * next_value * (1 - transition.done)
                    - transition.value
            )
            advantage = (
                    delta + self.gamma * self.gae_lambda * (1 - transition.done) * advantage
            )
            return (advantage, transition.value), advantage

        _, advantages = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_val), last_val),
            trajectories,
            reverse=True,
        )
        return advantages, advantages + trajectories.value

    def update(self, ts, batch):
        def loss_fn(params):
            _, log_prob, entropy, value = self.agent.apply(params, batch.trajectories.obs, None, batch.trajectories.action)

            # Actor loss
            entropy = entropy.mean()
            ratio = jnp.exp(log_prob - batch.trajectories.log_prob)
            advantages = (batch.advantages - batch.advantages.mean()) / (
                    batch.advantages.std() + 1e-8
            )
            clipped_ratio = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            pi_loss1 = ratio * advantages
            pi_loss2 = clipped_ratio * advantages
            pi_loss = -jnp.minimum(pi_loss1, pi_loss2).mean()
            actor_loss = pi_loss - self.ent_coef * entropy

            # Value loss
            value_pred_clipped = batch.trajectories.value + (
                    value - batch.trajectories.value
            ).clip(-self.clip_eps, self.clip_eps)
            value_losses = jnp.square(value - batch.targets)
            value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            value_loss =  self.vf_coef * value_loss

            return actor_loss + value_loss

        grads = jax.grad(loss_fn)(ts.agent_ts.params)
        return ts.replace(agent_ts=ts.agent_ts.apply_gradients(grads=grads))