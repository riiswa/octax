from octax import EmulatorState

rom_file = "UFO [Lutz V, 1992].ch8"

def score_fn(state: EmulatorState) -> float:
    return state.V[7]


def terminated_fn(state: EmulatorState) -> bool:
    return state.V[8] == 0

action_set = [4, 5, 6]

startup_instructions = 50

metadata = {
    "title": "UFO",
    "description": "Here's how to play UFO:\n\nYou have a stationary missle launcher at the bottom of the screen. You\ncan shoot in three directions; left diagonal, straight up, and right\ndiagonal.. using the keys 4, 5, and 6 respectively.. You try to hit\none of two objects flying by.. at apparently varying speeds..  Your\nscore is displayed on your left, the number of missles you have left\nis displayed on your right. (You get 15)..\n\nThis game (\"UFO\") is not new.  I have a copy of it from 1977 (!).  It\nwas one of the original CHIP-8 games on the audio cassette that was\nincluded when I bought my first computer, the Finnish-made Telmac\n1800.\n\nIt was also the first real program to run under CHIP-48 (it was used\nas a test case during the development of the CHIP-48 interpreter). The\nreason I have not posted it to the net myself is that I have no idea\nabout its copyright status.  I don't even know where it originated\n(RCA, perhaps?).\n\nThe cassette that was bundled with the Telmac 1800 contains more than\na dozen CHIP-8 programs.  If someone could convince me that these\nprograms are indeed freely redistributable, the other programs could\nalso be posted.  Otherwise, perhaps this one shouldn't have been.\n",
    "release": "1992",
    "authors": ["Lutz V"],
    "roms": {
      "bdb92475acfe11bc7814a2f5eade13fcd09b756a": {
        "file": "UFO [Lutz V, 1992].ch8",
        "platforms": ["originalChip8"]
      }
    }
  }