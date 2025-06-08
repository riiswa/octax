from octax import EmulatorState

rom_file = "Tank.ch8"

def score_fn(state: EmulatorState) -> float:
    return state.V[14]


def terminated_fn(state: EmulatorState) -> bool:
    return state.V[6] == 0

action_set = [2, 4, 5, 6, 8]

startup_instructions = 100

disable_delay = True

metadata = {
    "title": "Tank Battle",
    "description": "You are in a tank which has 25 bombs. Your goal is to hit 25 times a mobile target. The game ends when all your bombs are shot. If your tank hits the target, you lose 5 bombs. Use 2 4 6 and 8 to move. This game uses the original CHIP8 keyboard, so directions 2 and 8 are swapped.",
    "release": "197x",
    "roms": {
      "18b9d15f4c159e1f0ed58c2d8ec1d89325d3a3b6": {
        "file": "Tank.ch8",
        "platforms": ["originalChip8"]
      }
    }
  }