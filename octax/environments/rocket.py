from octax import EmulatorState
from octax.environments.brix import action_set

rom_file = "Rocket [Joseph Weisbecker, 1978].ch8"


def score_fn(state: EmulatorState) -> float:
    return state.V[1]


def terminated_fn(state: EmulatorState) -> bool:
    return state.V[2] == 9

action_set = [15]



metadata = {
    "title": "Rocket",
    "release": "1978-12",
    "authors": ["Joseph Weisbecker"],
    "description": "Rocket - by Joseph Weisbecker (1978)\n------------------------------------\nfrom Byte Magazine Dec.1978\n\n\nAn enemy UFO will be constantly moving from left to right across the top of the screen.\nA single digit score will be displayed at the lower right. A rocket ship will appear at \na random horizontal position along the bottom edge of the display area. You can launch\nthis rocket by pressing \"F\" on the hexadecimal keyboard.\n\nThe rocket will then move vertically toward the top of the screen. When it reaches the\ntop or hits the target UFO it will be erased and a new rocket will appear at the bottom\nof the screen.\n\nAfter nine rockets have been launched the game ends and no new rockets will appear. If\nyou hit the UFO with a rocket the score will incremented by 1.\n\n\nControls:\n---------\nF = launch rocket\n\n",
    "roms": {
      "3d1d029d6e31206d245c0ba881c0d1f003953bad": {
        "file": "Rocket [Joseph Weisbecker, 1978].ch8",
        "platforms": ["originalChip8"]
      }
    }
  }