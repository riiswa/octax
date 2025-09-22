from octax import EmulatorState
import jax.numpy as jnp

rom_file = "Airplane.ch8"

def score_fn(state: EmulatorState) -> float:
    return -jnp.astype(state.V[11], jnp.int32)


def terminated_fn(state: EmulatorState) -> bool:
    return (state.V[11] == 0)| (state.V[12] == 6)

action_set = [8]

startup_instructions = 600

disable_delay = True

metadata = {
    "title": "Airplane",
    "description": "Airplane - Blitz type of bombing game. Hit \"8\" to drop a bomb.",
    "release": "19xx",
    "roms": {
      "fca71182a8838b686573e69b22aff945d79fe1d0": {
        "file": "Airplane.ch8",
        "platforms": ["originalChip8"]
      }
    }
  }