from octax import EmulatorState
import jax.numpy as jnp

rom_file = "Airplane.ch8"

# Reward hitting targets (decrease V[11]) and penalize level progression (increase V[12])
def score_fn(state: EmulatorState) -> float:
    return -state.V[11] - state.V[12]


def terminated_fn(state: EmulatorState) -> bool:
    return (state.V[11] == 0)| (state.V[12] == 6)

action_set = [8]

startup_instructions = 600

disable_delay = False

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