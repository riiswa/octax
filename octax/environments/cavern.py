"""
MODIFIED ROM: Cavern with Leftward Progress Reward
================================================

Original: Cavern by Matthew Mikolay (2014) - Cave escape game
Modified: Added leftward exploration reward system for RL training

Key Changes:
- Removed menu/level selection
- Replaced survival scoring with leftward progress reward
- V[0]: Non-decreasing score (+1 for each new leftmost X position reached)
- V[A]: Tracks leftmost X position ever visited
- V[E]: Game state flag (1=playing, 0=game over)
- V[B], V[C]: Temporary registers for position comparison logic
- Goal: Encourage agent to explore leftward (lower X coordinates)
- Reward only given for reaching previously unvisited leftmost positions

"""

from octax import EmulatorState

# Defined in __init__: rom_file = "cavern1.ch8"

def score_fn(state: EmulatorState) -> float:
    return state.V[0]


def terminated_fn(state: EmulatorState) -> bool:
    return state.V[14] == 0


action_set = [2, 4, 6, 8]

startup_instructions = 250

metadata = {
    "title": "Cavern",
    "release": "2014",
    "authors": ["Matthew Mikolay"],
    "description": "\nCavern - by Matthew Mikolay (2014)\n----------------------------------\n\nHello all!\n\nI recently finished a new game implemented in CHIP8. It's called Cavern, and is conceptually based \nupon the CHIP8 game Cave.\n\nCavern allows the player to select one of three speeds for gameplay, though the easiest setting is \nstill pretty challenging. I suppose Cavern could be considered a game suitable for individuals who \nfound Cave too easy.\n\nThe in-game controls are 2, 4, 6, and 8 to move the sprite on screen.\n\nI've attached the .ch8 file for anyone interested in playing. It has been tested on the Emma emulator. \nIf anybody comes across any bugs, please let me know!\n\nBest,\nMatt\n\n----------------------------------------------------------\n\nEscape the cave without crashing into the walls!\n\n\nNavigate using the '2', '4', '6', and '8' keys.\n\n----------------------------------------------------------\n",
    "roms": {
      "17238bcd1cb8e21142a1d7533f878c833ef19caa": {
        "file": "Cavern (by Matthew Mikolay)(2014).ch8",
        "platforms": ["originalChip8"]
      }
    }
  }