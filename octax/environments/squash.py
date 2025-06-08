from octax import EmulatorState

rom_file = "Squash [David Winter].ch8"

def score_fn(state: EmulatorState) -> float:
    return state.V[11]

def terminated_fn(state: EmulatorState) -> bool:
    return state.V[11] == 0

action_set = [1, 4]

startup_instructions = 100

metadata = {
    "title": "Squash",
    "description": "Squash, by David Winter\n\nBounce a ball around a squash court with your paddle",
    "release": "1997",
    "authors": ["David Winter"],
    "roms": {
      "a58ec7cc63707f9e7274026de27c15ec1d9945bd": {
        "file": "Squash [David Winter].ch8",
        "platforms": ["originalChip8", "modernChip8"],
        "keys": {
          "up": 1,
          "down": 4
        }
      }
    }
  }