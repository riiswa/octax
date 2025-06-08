"""BRIX - Breakout clone

Control paddle to bounce ball and destroy bricks.
Score: V5 (increments per brick destroyed)
Lives: V14 (starts at 3, game over when 0)
Controls: Left/Right paddle movement
"""

from octax import EmulatorState

rom_file = "Brix [Andreas Gustafsson, 1990].ch8"


def score_fn(state: EmulatorState) -> float:
    """Score increases when bricks are destroyed."""
    return state.V[5]


def terminated_fn(state: EmulatorState) -> bool:
    """Game ends when lives (V14) reach zero."""
    return state.V[14] == 0


action_set = [4, 6]  # Left, Right

startup_instructions = 500

disable_delay = True

metadata = {
    "title": "Brix",
    "description": "\nBrix - by Andreas Gustafsson (1990)\n-----------------------------------\n\nThis game is an \"arkanoid\" precursor. You have 5 lives, and your\ngoal is the destruction of all the brixs. Use 4 and 6 to move\nyour paddle. The game ends when all the brixs are destroyed.\n\n\n",
    "release": "1990",
    "authors": ["Andreas Gustafsson"],
    "roms": {
      "f13766c14aeb02ad8d4d103cb5eadd282d20cddc": {
        "file": "Brix [Andreas Gustafsson, 1990].ch8",
        "platforms": ["originalChip8"]
      }
    }
  }