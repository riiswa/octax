"""CHIP-8 emulator state structures."""

import jax
import jax.numpy as jnp
from flax.struct import dataclass, PyTreeNode, field

from octax.constants import PROGRAM_START, FONT_START, FONT_DATA, SCREEN_WIDTH, SCREEN_HEIGHT, STACK_SIZE


@dataclass(frozen=True)
class StackState:
    """Stack state for subroutine calls."""
    data: jnp.ndarray = jnp.zeros(STACK_SIZE, dtype=jnp.uint16)
    pointer: int = 0


class EmulatorState(PyTreeNode):
    """Main CHIP-8 emulator state."""
    rng: jax.random.PRNGKey
    memory: jnp.ndarray = jnp.zeros(4096, dtype=jnp.uint8)
    pc: jnp.ndarray = jnp.astype(PROGRAM_START, jnp.uint16)
    display: jnp.ndarray = jnp.zeros((SCREEN_WIDTH, SCREEN_HEIGHT), dtype=jnp.bool_)
    stack: StackState = StackState()
    delay_timer: jnp.ndarray = jnp.zeros((), dtype=jnp.uint8)
    sound_timer: jnp.ndarray = jnp.zeros((), dtype=jnp.uint8)
    keypad: jnp.ndarray = jnp.zeros(16, dtype=jnp.bool_)
    V: jnp.ndarray = jnp.zeros(16, dtype=jnp.uint8)
    I: jnp.ndarray = jnp.zeros((), dtype=jnp.uint16)
    modern_mode: bool = field(pytree_node=False, default=True)


def create_state(rng: jax.random.PRNGKey = jax.random.PRNGKey(0)) -> EmulatorState:
    """Create initial emulator state with font data loaded."""
    state = EmulatorState(rng)
    return state.replace(memory=state.memory.at[FONT_START:FONT_START + len(FONT_DATA)].set(FONT_DATA))
