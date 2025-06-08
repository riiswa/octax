from octax import EmulatorState
from octax.env import run_n_instruction

rom_file = "Vertical Brix [Paul Robson, 1996].ch8"

def score_fn(state: EmulatorState) -> float:
    return state.V[8]


def terminated_fn(state: EmulatorState) -> bool:
    """Game ends when lives (V14) reach zero."""
    return state.V[7] == 0

def custom_startup(state: EmulatorState):
    state = state.replace(keypad=state.keypad.at[7].set(1))
    state = run_n_instruction(state, 1000)
    state = state.replace(keypad=state.keypad.at[7].set(0))
    return state


action_set = [1, 4]

disable_delay = True

metadata = {
    "title": "Vertical Brix",
    "description": "\nVertical Brix - by Paul Robson (1996)\n-------------------------------------\n\nLike BRIX, but the brix are put vertically, and the pad also moves vertically. \n\nStart by pressing 7, and move using 4 and 1.\n\n*NOTE: 7->5 1->2 4->6",
    "release": "1996",
    "authors": ["Paul Robson"],
    "roms": {
      "da710f631f8e35534d0b9170bcf892a60f49c43d": {
        "file": "Vertical Brix [Paul Robson, 1996].ch8",
        "platforms": ["originalChip8"]
      }
    }
  }