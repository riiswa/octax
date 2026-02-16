import pickle
import time
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from octax.agents import PPOOctax, PQNOctax
from octax.environments import create_environment
from octax.wrappers import OctaxGymnaxWrapper
from rejax.evaluate import evaluate

import jax
import jax.numpy as jnp


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg)

    env_name = cfg.pop("env", None)

    env, metadata = create_environment(env_name)
    env = OctaxGymnaxWrapper(env)
    env_params = env.default_params

    job_id = str(time.strftime("%Y%m%d_%H%M%S") + "_" + "__".join(
        [str(k) + "_" + str(v).lower().replace(" ", "_") for k, v in cfg.items() if
         isinstance(v, (str, int, float, bool, type(None)))]
    ))

    def eval_callback(algo, ts, rng):
        act = algo.make_act(ts)
        max_steps = algo.env_params.max_steps_in_episode
        lengths, returns = evaluate(act, rng, env, env_params, 128, max_steps)

        jax.debug.print(
            "Step {}, Mean episode length: {}, Mean return: {}",
            ts.global_step,
            lengths.mean(),
            returns.mean(),
        )
        return lengths, returns

    rng = jax.random.PRNGKey(cfg.pop("seed", 0))
    num_seeds = cfg.pop("num_seeds", 1)
    rngs = jax.random.split(rng, num_seeds)

    agent_name = cfg.pop("agent", None)
    if agent_name is not None:
        if agent_name == "PPO":
            agent_class = PPOOctax
        elif agent_name == "PQN":
            agent_class = PQNOctax
        else:
            raise ValueError(f"Unknown agent name: {agent_name}")

    agent = PPOOctax.create_agent(cfg, env, env_params)
    algo = PPOOctax(env=env, env_params=env_params, agent=agent, eval_callback=eval_callback, **cfg)

    start = time.time()
    vmap_train = jax.jit(jax.vmap(algo.train))
    ts, (_, returns) = vmap_train(rngs)
    returns.block_until_ready()
    end = time.time()

    t = jnp.arange(returns.shape[1]) * algo.eval_freq
    returns = returns.mean(axis=-1)

    os.makedirs(f"results/{env_name}/", exist_ok=True)

    params = []

    with open(f"results/{env_name}/{job_id}.pkl", 'wb') as f:
        pickle.dump(
            {"params": params,"timesteps": t, "returns": returns, "config": cfg, "time": end-start},
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    with open("results/results.txt", "a") as f:
        f.write(f"{env_name}__{job_id}\t{returns[:, -1].mean()}±{returns[:, -1].std()}\n")

    print(f"Results saved in results/{env_name}/{job_id}.pkl")


if __name__ == "__main__":
   main()
