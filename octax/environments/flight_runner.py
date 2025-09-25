from octax import EmulatorState

rom_file = "flightrunner.ch8"

def score_fn(state: EmulatorState) -> float:
    return state.V[7]

def terminated_fn(state: EmulatorState) -> bool:
    return (state.V[5] == 255) | (state.V[7] == 255)

action_set = [5, 7, 8, 9]

startup_instructions = 500

metadata = {
    "title": "Flight Runner",
    "authors": ["TodPunk"],
    "images": ["flightrunner.gif"],
    "release": "2014-11-01",
    "description": "A simple flight runner for the Chip8.",
    "roms": {
      "821751787374cc362f4c58759961f0aa7a2fd410": {
        "file": "flightrunner.ch8",
        "platforms": ["modernChip8"],
        "tickrate": 15,
        "colors": {
          "pixels": ["#0000ff", "#ffcc00"],
          "buzzer": "#ffffff",
          "silence": "#000000"
        }
      }
    },
    "origin": {
      "type": "gamejam",
      "reference": "Octojam1"
    }
  }