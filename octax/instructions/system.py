"""CHIP-8 system instructions (0x0xxx)."""

import jax
import jax.lax
import jax.numpy as jnp
from octax.state import EmulatorState
from octax.decode import DecodedInstruction
from octax.stack import pop


def no_op(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """No operation."""
    return state


def execute_clear_screen(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """00E0 - Clear display."""
    return state.replace(display=jnp.empty_like(state.display))


def execute_return(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """00EE - Return from subroutine."""
    stack, address = pop(state.stack)
    return state.replace(stack=stack, pc=address)


def execute_system_instruction(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """Dispatch system instructions."""
    return jax.lax.cond(
        0x00E0 == instruction.raw,
        execute_clear_screen,
        lambda state, instruction: jax.lax.cond(
            0x00EE == instruction.raw,
            execute_return,
            no_op,
            state, instruction
        ),
        state, instruction
    )