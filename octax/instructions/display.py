"""CHIP-8 display operations."""

import jax.numpy as jnp
from octax.state import EmulatorState
from octax.decode import DecodedInstruction
from octax.constants import SCREEN_WIDTH, SCREEN_HEIGHT

# Pre-computed coordinate grids for display operations
xx, yy = jnp.meshgrid(jnp.arange(SCREEN_WIDTH), jnp.arange(SCREEN_HEIGHT), indexing='ij')


def execute_display(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """DXYN - Draw sprite at (VX, VY) with height N."""
    sprite_x = state.V[instruction.x] % SCREEN_WIDTH
    sprite_y = state.V[instruction.y] % SCREEN_HEIGHT

    in_screen = (xx >= sprite_x) & (xx < sprite_x + 8) & (yy >= sprite_y) & (yy < sprite_y + instruction.n)

    row_offset = yy - sprite_y
    col_offset = xx - sprite_x
    sprite_bytes = state.memory[state.I + row_offset]
    sprite = (sprite_bytes >> (7 - col_offset)) & 1 & in_screen

    return state.replace(
        display=jnp.astype(state.display ^ sprite, jnp.bool_),
        V=state.V.at[15].set(jnp.any(state.display & sprite))
    )