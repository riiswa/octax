from octax import EmulatorState
from octax.env import run_n_instruction

rom_file = "Deep8 (by John Earnest)(2014).ch8"


def score_fn(state: EmulatorState) -> float:
    return state.V[9]


def terminated_fn(state: EmulatorState) -> bool:
    return state.V[0xB] != 1

def custom_startup(state: EmulatorState):
    state = state.replace(keypad=state.keypad.at[0].set(1))
    state = run_n_instruction(state, 150)
    state = state.replace(keypad=state.keypad.at[0].set(0))
    return state


action_set = [7, 8, 9]


metadata = {
    "title": "Deep8",
    "release": "2014",
    "authors": ["John Earnest"],
    "description": "\nDeep8 - by John Earnest (2014)\n------------------------------\n\nA stripped down port of the Mako game \"Deep\" for Chip8. Move your boat \nleft and right with A/D. Press S to drop a bomb and release S to detonate \nit. Destroy incoming squid before they tip your boat!\n\nThis game was one of the earliest programs written using Octo.\n\n\n\n",
    "roms": {
      "b41cc0b5b2faabafd532d705b804abb3e8f97baf": {
        "file": "Deep8 (by John Earnest)(2014).ch8",
        "platforms": ["originalChip8"]
      }
    }
  }