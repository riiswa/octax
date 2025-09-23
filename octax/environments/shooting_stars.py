import jax
import jax.numpy as jnp

from octax import EmulatorState


rom_file = "Shooting Stars [Philip Baltzer, 1978].ch8"

def score_fn(state: EmulatorState) -> float:
    return jax.lax.cond(state.V[0] > 128, lambda: jnp.astype(0, jnp.uint8),  lambda: state.V[0])


def terminated_fn(state: EmulatorState) -> bool:
    return False

action_set = [2, 8, 4, 6]

disable_delay = True

metadata = {
    "title": "Shooting Stars",
    "description": "Shooting Stars (1978), by Philip Baltzer",
    "authors": ["Philip Baltzer"],
    "release": "1978",
    "roms": {
      "443550abf646bc7f475ef0466f8e1232ec7474f3": {
        "file": "Shooting Stars [Philip Baltzer, 1978].ch8",
        "platforms": ["originalChip8"]
      }
    }
  }