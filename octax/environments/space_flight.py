"""
MODIFIED ROM: Space Flight - Single Life Mode
===================================================

Original: Space Flight (19xx) - Multi-level with 3 lives
Modified: Single level, single life, immediate termination on collision

Key Changes:
- Single life only (V[9] = 1)
- Immediate game over on collision
- V[0]: Distance score (+1 per frame survived)
- V[9]: Lives (1 → 0 on collision)
- V[E]: Game state flag (1=playing, 0=game over)

Goal: Navigate from left to right avoiding asteroids in single attempt
"""

from octax import EmulatorState


def score_fn(state: EmulatorState) -> float:
    return state.V[0]


def terminated_fn(state: EmulatorState) -> bool:
    """Game ends when lives (V14) reach zero."""
    return (state.V[9] == 0) | (state.V[12] >= 0x3E)

action_set = [1, 4]

startup_instructions = 300

metadata = {
    "title": "Space Flight",
    "description": "Space flight game\n</br>Fly through the asteroid field. Use 1 and 4 key to navigate space ship and E/F to start the game.",
    "release": "19xx",
    "roms": {
      "aa4f1a282bd64a2364102abf5737a4205365a2b4": {
        "file": "Space Flight.ch8",
        "platforms": ["originalChip8"]
      }
    }
  }