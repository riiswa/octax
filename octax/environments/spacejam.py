from octax import EmulatorState

rom_file = "spacejam.ch8"

def score_fn(state: EmulatorState) -> float:
    return state.V[9]


def terminated_fn(state: EmulatorState) -> bool:
    return state.V[10] == 0

action_set = [5, 8, 7, 9]

disable_delay = True

startup_instructions = 300

metadata = {
    "title": "Spacejam!",
    "authors": ["WilliamDonnelly"],
    "images": ["title.gif"],
    "release": "2015-10-30",
    "description": "An extended and enhanced version loosely based on ShipTunnel from the 2014 OctoJam.",
    "roms": {
      "9f9a4affbf7afd70bb594fb321e16579318c0164": {
        "file": "spacejam.ch8",
        "platforms": ["modernChip8"],
        "tickrate": 100,
        "colors": {
          "pixels": ["#111111", "#fcfcfc"],
          "buzzer": "#ffaa00",
          "silence": "#000000"
        }
      }
    },
    "origin": {
      "type": "gamejam",
      "reference": "Octojam2"
    }
  }