"""CHIP-8 emulator package."""

from octax.state import EmulatorState, create_state
from octax.emulator import execute, load_rom, fetch
from octax.decode import DecodedInstruction, decode
from octax.constants import *
from octax.rendering import chip8_display_to_rgb, create_color_scheme, batch_render

__all__ = [
    "EmulatorState",
    "create_state",
    "fetch",
    "execute",
    "load_rom",
    "DecodedInstruction",
    "decode",
    "PROGRAM_START",
    "FONT_START",
    "SCREEN_WIDTH",
    "SCREEN_HEIGHT",
    "chip8_display_to_rgb",
    "create_color_scheme",
    "batch_render",
]
