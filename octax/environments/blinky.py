from octax import EmulatorState

rom_file = "Blinky [Hans Christian Egeberg, 1991].ch8"

def score_fn(state: EmulatorState) -> float:
    return state.V[6]


def terminated_fn(state: EmulatorState) -> bool:
    return state.V[3] == 255

action_set = [3, 6, 7, 8]

startup_instructions = 13800

metadata = {
    "title": "Blinky",
    "release": "1991",
    "authors": ["Hans Christian Egeberg"],
    "description": "Blinky has two lives. Finishing a screen increases the intelligence\nof Packlett and Heward, the two ghosts. The game is at its most difficult\non the fourth screen. The maze contains one gateway from left to right,\nand four energy pills, near the corners. Points are awarded for each pill,\neach energy pill, catching Packlett, catching Heward, and finishing a maze.\n\nMy personal highscore is 1575 so far, but I am just a programmer.\nMy not so little sister helped me debug, and reached 2005 in two days.",
    "roms": {
      "5b733a60e7208f6aa0d15c99390ce4f670b2b886": {
        "file": "Blinky (by Hans Christian Egeberg)(1991).sc8",
        "platforms": ["superchip"],
        "description": "Blinky V2.00: Pac Man game for SCHIP V1.0 or newer.\nFrom: egeberg@solan.unit.no (Hans Christian Egeberg)\n\nThis program is for use with Super Chip-48 V1.0 or newer.\nIn order to run, put downloaded binary string on stack, and start SCHIP.",
        "embeddedTitle": "2.00 C. Egeberg 18/8-'91"
      },
      "5370ecf9ae444c71b63dab9b1f9968a4fe67c9dd": {
        "file": "Blinky (fix)[Hans Christian Egeberg, 1991].ch8",
        "platforms": ["modernChip8"],
        "description": "This version does not have the Fx55/Fx65 (load_store_quirk) or shift issue (where map is rendered \nimproperly and it freezes when you hit a wall)\n\n\n\nThis game is a PACMAN clone. \nYour goal is to eat all the balls in the maze. There are some enemies, so be careful. \n\nUse 2, 4, 6 and 8 to move.\n",
        "embeddedTitle": "2.00 C. Egeberg 18/8-'91"
      },
      "d40abc54374e4343639f993e897e00904ddf85d9": {
        "file": "Blinky [Hans Christian Egeberg, 1991].ch8",
        "platforms": ["superchip"],
        "description": "Blinky (1991), by Hans Christian Egeberg\n\nPacman clone.\n3, 6 - down/up. 7, 8 - left/right",
        "embeddedTitle": "2.00 C. Egeberg 18/8-'91"
      },
      "f4169141735d8d60e51409ca7e73f4adedcefef2": {
        "file": "Blinky [Hans Christian Egeberg] (alt).ch8",
        "platforms": ["superchip"]
      }
    }
  }