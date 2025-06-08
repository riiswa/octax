from octax import EmulatorState

rom_file = "Submarine [Carmelo Cortez, 1978].ch8"

def score_fn(state: EmulatorState) -> float:
    return state.V[7]


def terminated_fn(state: EmulatorState) -> bool:
    return state.V[8] == 0

action_set = [5]

startup_instructions = 100


metadata = {
    "title": "Submarine",
    "description": "Submarine (1978), by Carmelo Cortez\n\nThe Sub Game is my favorite. Press \"5\" key to fire depth charges at the subs below. You score 15 points for a small sub and 5 points for the larger. You get 25 depth charges to start.",
    "authors": ["Carmelo Cortez"],
    "roms": {
      "89aadf7c28bcd1c11e71ad9bd6eeaf0e7be474f3": {
        "file": "Submarine [Carmelo Cortez, 1978].ch8",
        "platforms": ["originalChip8"]
      }
    }
  }