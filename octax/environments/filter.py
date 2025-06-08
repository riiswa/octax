from octax import EmulatorState

rom_file = "Filter.ch8"

def score_fn(state: EmulatorState) -> float:
    return state.V[14]


def terminated_fn(state: EmulatorState) -> bool:
    return state.V[13] == 0

action_set = [4, 6]


metadata = {
    "title": "Filter",
    "description": "Filter\n\nCatch the drop coming from the pipe at the top of the screen with your paddle.",
    "roms": {
      "ae71a7b081a947f1760cdc147759803aea45e751": {
        "file": "Filter.ch8",
        "platforms": ["originalChip8"]
      }
    }
  }