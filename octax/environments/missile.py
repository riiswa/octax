from octax import EmulatorState

rom_file = "Missile [David Winter].ch8"


def score_fn(state: EmulatorState) -> float:
    return state.V[7]

def terminated_fn(state: EmulatorState) -> bool:
    return state.V[6] == 0

action_set = [8]  # Shoot

startup_instructions = 50

metadata = {
    "title": "Missile Command",
    "description": "\nMissile Command - by David Winter (19xx)\n----------------------------------------\n\nYou must shoot the 8 targets on the screen using key 8. \n\nYour shooter moves a little bit faster each time you shoot. \nYou have 12 missiles to shoot all the targets, and you win 5\npoints per target shot.\n\n\n",
    "release": "1996",
    "authors": ["David Winter"],
    "roms": {
      "0d0cc129dad3c45ba672f85fec71a668232212cc": {
        "file": "Missile [David Winter].ch8",
        "platforms": ["originalChip8", "modernChip8"],
        "embeddedTitle": "MISSILE by David WINTER",
        "keys": {
          "a": 8
        }
      }
    }
  }

