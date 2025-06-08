from octax import EmulatorState
import jax.numpy as jnp


rom_file = "Pong (1 player).ch8"

def score_fn(state: EmulatorState) -> float:
    return (jnp.astype(state.V[14], jnp.int32) // 10) - (jnp.astype(state.V[14], jnp.int32) % 10)


def terminated_fn(state: EmulatorState) -> bool:
    return ((state.V[14] // 10) == 9) | ((state.V[14] % 10) == 9)

action_set = [1, 4]

disable_delay = True

metadata = {
    "title": "Pong",
    "description": "Single player pong game",
    "authors": ["1 player"],
    "roms": {
      "607c4f7f4e4dce9f99d96b3182bfe7e88bb090ee": {
        "file": "Pong (1 player).ch8",
        "platforms": ["originalChip8"]
      }
    }
  }