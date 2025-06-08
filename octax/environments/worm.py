import jax

from octax import EmulatorState

rom_file = "Worm V4 [RB-Revival Studios, 2007].ch8"

def score_fn(state: EmulatorState) -> float:
    return state.V[5]


def terminated_fn(state: EmulatorState) -> bool:
    return state.V[7] == 255  #(state.V[10] == 58) | (state.V[10] == 254) | (state.V[11] == 30) | (state.V[11] == 254) | (state.V[11] == 30)

action_set = [2, 8, 4, 6]

startup_instructions = 1600

metadata = {
    "title": "SuperWorm V4",
    "description": "-----------------------------------------------------------------------------\n\t\t\t      /////////////////\n\t                      //////////////////\n        \t              ////          ////\n\t               \t      ////   ///////////\n\t                      ////  ///////////\n                              ////  ////\n                              ////  ///////////\n                              ////   //////////\n  \t     \t   \t\n  \t\t\t   www.revival-studios.com\n-----------------------------------------------------------------------------\nTitle\t\t:\tSuperWorm V4\nAuthor\t\t:\tRB (Original game)\n\t\t \tUpdates and fixes by: Martijn Wenting / Revival Studios\nGenre\t\t:\tGame\nSystem\t\t:\tChip-8 / SuperChip8\nDate\t\t:\t10/08/2007 \nProduct ID\t:\tRS-C8001\n-----------------------------------------------------------------------------\n\nAll the contents of this package are (c)Copyright 2007 Revival Studios.\nOriginal game: SuperWorm is (c)Copyright 1992 RB\n\nThe contents of the package may only be spread in its original form, and may not be\npublished or distributed otherwise without the written permission of the authors.\n\nDescription:\n------------\nSuperWorm V4 is an update of the SuperChip8 game: Worm3 by RB.\nThe original game was only for SuperChip, so i've created a Chip-8 port.\nIt also includes several speed fixes and a new control system.\n\nRunning the game:\n-----------------\nUse the Megachip emulator or any other Chip-8/SuperChip compatible emulator to run the game.\n\nCredits:\n--------\nChip-8 version, Updates and fixes by: Martijn Wenting\nOriginal game by: RB\n\nDistribution:\n-------------\nThis package can be freely distributed in its original form.\nIf you would like to include this game in your rom package, please let me know.\n\nWatch out for more releases soon!\n\n\n\tMartijn Wenting / Revival Studios\n\n",
    "release": "2007",
    "authors": ["RB-Revival Studios", "Martijn Wenting"],
    "roms": {
      "a1c1e0e7b01004be3ee77c69030e6b536cb316e6": {
        "file": "Worm V4 [RB-Revival Studios, 2007].ch8",
        "platforms": ["originalChip8"]
      },
      "2d415bf1f31777b22ad73208c4d1ad27d5d4f367": {
        "file": "SuperWorm V4 (by RB & Revival Studios)(2007).sc8",
        "platforms": ["superchip"]
      }
    }
  }