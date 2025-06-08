from octax import EmulatorState
from octax.environments.brix import action_set

rom_file = "Tetris [Fran Dachille, 1991].ch8"


def score_fn(state: EmulatorState) -> float:
    return state.V[10]

def terminated_fn(state: EmulatorState) -> bool:
    return state.V[1] == 2

action_set = [4, 5, 6, 7]

startup_instructions = 300

metadata = {
    "title": "Tetris",
    "release": "1991",
    "authors": ["Fran Dachille"],
    "description": "                               TETRIS\n                          by Fran Dachille\n\nThis is my first release of the famous Tetris game on the HP48S. I was\ninspired by the lack enjoyable games for our favorite handheld.  [Not since the\nGoodies Disks have been available!  -jkh-]  This game, though it lacks some of\nthe whistles and bangs of fancy versions, performs on par with quality arcade\nversions (nota bene -> SPEED).  At my college, every person who picks up my\ncalculator is immediately hooked for hours.\n\nThis version is written for the CHIP48 game interpreter (c)\ncopyright 1990 Andreas Gustafsson.  \n\nThe 4 key is left rotate, 5 - left move, 6 - right move, 1\n- drop, ENTER - restart, DROP - end.  After every 5 lines, the speed\nincreases slightly and peaks at 45 lines.\n\nThere is room for improvement in this version.  Notably, background\npictures, a pause key (for now, hold ON), two rotate keys, various\nstarting skill levels, a B version which starts with randomn blocks,\nfinishing graphics, and high scores, just to name a few.\n\nIn order for improvements, I need to know if there is reasonable\ndemand.  If this game is worth playing for hours upon hours, please let\nme know.  If you wish to support the improvements, want future versions,\nand want to see other games ported to the HP48S, send $5.00 to:\n\n          FRAN DACHILLE\n          WEBB INSTITUTE\n          GLEN COVE, NY 11542\n\n",
    "roms": {
      "5f518084744bf3cb8733f6e5454dfd1634320563": {
        "file": "Tetris [Fran Dachille, 1991].ch8",
        "platforms": ["chip48", "originalChip8", "modernChip8"],
        "keys": {
          "left": 5,
          "right": 6,
          "down": 7,
          "a": 4
        }
      }
    }
  }