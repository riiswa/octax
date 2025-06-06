"""CHIP-8 control flow instructions."""

import jax
import jax.lax
import jax.numpy as jnp
from octax.state import EmulatorState
from octax.decode import DecodedInstruction
from octax.stack import push


def execute_jump(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """1NNN - Jump to address NNN."""
    return state.replace(pc=jnp.astype(instruction.nnn, jnp.uint16))


def execute_call(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """2NNN - Call subroutine at NNN."""
    state = state.replace(stack=push(state.stack, state.pc))
    return execute_jump(state, instruction)


def make_skip_instruction(condition_fn):
    """Factory for skip instructions."""
    def skip_instruction(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
        condition = condition_fn(state, instruction)
        return jax.lax.cond(
            condition,
            lambda s: s.replace(pc=s.pc + 2),
            lambda s: s,
            state
        )
    return skip_instruction


execute_skip_if_equal_immediate = make_skip_instruction(
    lambda state, inst: state.V[inst.x] == inst.nn
)

execute_skip_if_not_equal_immediate = make_skip_instruction(
    lambda state, inst: state.V[inst.x] != inst.nn
)

execute_skip_if_equal_register = make_skip_instruction(
    lambda state, inst: state.V[inst.x] == state.V[inst.y]
)

execute_skip_if_not_equal_register = make_skip_instruction(
    lambda state, inst: state.V[inst.x] != state.V[inst.y]
)


def execute_jump_with_offset_modern(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """BXNN - Jump to address NN + VX (modern behavior)."""
    address_offset = instruction.nn
    register_value = state.V[instruction.x]
    jump_address = (address_offset + jnp.astype(register_value, jnp.uint16)) & 0xFFF
    return state.replace(pc=jump_address)


def execute_jump_with_offset_legacy(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """BNNN - Jump to address NNN + V0 (legacy behavior)."""
    jump_address = (instruction.nnn + jnp.astype(state.V[0], jnp.uint16)) & 0xFFF
    return state.replace(pc=jump_address)


def execute_skip_if_key(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """EX9E/EXA1 - Skip if key pressed/not pressed."""

    key_index = state.V[instruction.x] & 0xF
    key_pressed = state.keypad[key_index]
    is_not_instruction = (instruction.nn == 0xA1)
    condition = key_pressed ^ is_not_instruction

    return jax.lax.cond(
        condition,
        lambda state: state.replace(pc=state.pc + 2),
        lambda state: state,
        state
    )