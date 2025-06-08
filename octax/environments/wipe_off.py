from octax import EmulatorState

rom_file = "Wipe Off [Joseph Weisbecker].ch8"

def score_fn(state: EmulatorState) -> float:
    return state.V[6]


def terminated_fn(state: EmulatorState) -> bool:
    return state.V[7] == 0

action_set = [4, 6]

startup_instructions = 500


metadata = {
    "title": "Wipe Off",
    "description": "\nWipe Off - by Joseph Weisbecker (19xx)\n--------------------------------------\nCosmac VIP CDP18S711 Book - Page 42 (VIP-311)\n\n\nThis program uses the CHIP-8 INTERPRETER at 0000-01FF Serve the ball by \npressing any key. Move the paddle left or right by pressing key 4 or 6. \n\nTry to wipe out as many spots as possible. Each spot counts one point. \nYou get 20 balls. You see your final score at the end of the game. You can \nmake the paddle wider by changing the E0 byte at 02CD to F8 or FF. ",
    "release": "19xx",
    "authors": ["Joseph Weisbecker"],
    "roms": {
      "d666688a8fce468a7d88b536bc1ef5f35ba12031": {
        "file": "Wipe Off [Joseph Weisbecker].ch8",
        "platforms": ["originalChip8"]
      }
    }
}
