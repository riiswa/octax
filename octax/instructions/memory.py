"""CHIP-8 memory and register operations."""

import jax
import jax.numpy as jnp
from octax.state import EmulatorState
from octax.decode import DecodedInstruction


def execute_set(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """6XNN - Set VX = NN."""
    return state.replace(V=state.V.at[instruction.x].set(instruction.nn))


def execute_add(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """7XNN - Add NN to VX."""
    return state.replace(V=state.V.at[instruction.x].add(instruction.nn))


def execute_set_index(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """ANNN - Set I = NNN."""
    return state.replace(I=jnp.astype(instruction.nnn, jnp.uint16))


def execute_random(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """CXNN - Set VX = random & NN."""
    key, subkey = jax.random.split(state.rng)
    random_value = jax.random.randint(subkey, shape=(), minval=0, maxval=256, dtype=jnp.uint8)
    return state.replace(V=state.V.at[instruction.x].set(random_value & instruction.nn), rng=key)