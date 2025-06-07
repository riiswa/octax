"""CHIP-8 miscellaneous instructions (Fxxx)."""

import jax
import jax.lax
import jax.numpy as jnp
from octax.state import EmulatorState
from octax.decode import DecodedInstruction
from octax.constants import FONT_START
from octax.instructions.system import no_op


def execute_get_delay_timer(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX07 - Set VX to delay timer value."""
    return state.replace(V=state.V.at[instruction.x].set(state.delay_timer))


def execute_set_delay_timer(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX15 - Set delay timer to VX."""
    return state.replace(delay_timer=state.V[instruction.x])


def execute_set_sound_timer(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX18 - Set sound timer to VX."""
    return state.replace(sound_timer=state.V[instruction.x])


def execute_add_to_index(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX1E - Add VX to I register."""
    new_i = jnp.astype(state.I + state.V[instruction.x], jnp.uint16)
    overflow_flag = new_i > 0xFFF
    new_vf = jnp.where(overflow_flag, 1, 0)
    return state.replace(
        I=new_i & 0xFFF,
        V=state.V.at[15].set(new_vf)
    )


def execute_wait_for_key(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX0A - Wait for key press (blocking)."""
    def key_pressed_action(state):
        pressed_key = jnp.argmax(state.keypad)
        return state.replace(V=state.V.at[instruction.x].set(pressed_key))

    def wait_action(state):
        return state.replace(pc=state.pc - 2)

    return jax.lax.cond(jnp.any(state.keypad), key_pressed_action, wait_action, state)


def execute_font_character(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX29 - Set I to location of sprite for digit VX."""
    font_address = FONT_START + (state.V[instruction.x] * 5)
    return state.replace(I=jnp.astype(font_address, jnp.uint16))


def execute_bcd_conversion(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX33 - Store BCD representation of VX at I, I+1, I+2."""
    value = state.V[instruction.x]

    # Vectorized BCD conversion
    digits = jnp.array([
        value // 100,
        (value // 10) % 10,
        value % 10
    ], dtype=jnp.uint8)

    # Single vectorized memory update
    indices = jnp.arange(3) + state.I
    new_memory = state.memory.at[indices].set(digits)
    return state.replace(memory=new_memory)


def execute_store_registers(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX55 - Store V0 through VX in memory starting at I."""
    register_mask = jnp.arange(16) <= instruction.x
    base_indices = state.I + jnp.arange(16)
    current_memory_values = state.memory[base_indices]
    new_memory_values = jnp.where(register_mask, state.V, current_memory_values)
    new_memory = state.memory.at[base_indices].set(new_memory_values)

    if state.modern_mode:
        return state.replace(memory=new_memory)
    else:
        return state.replace(memory=new_memory, I=state.I + instruction.x + 1)


def execute_load_registers(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """FX65 - Load V0 through VX from memory starting at I."""
    register_mask = jnp.arange(16) <= instruction.x
    base_indices = state.I + jnp.arange(16)
    memory_values = state.memory[base_indices]
    new_V = jnp.where(register_mask, memory_values, state.V)

    if state.modern_mode:
        return state.replace(V=new_V)
    else:
        return state.replace(V=new_V, I=state.I + instruction.x + 1)


def execute_misc_instruction(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """Dispatch misc instructions using arithmetic switch."""
    is_0x07 = instruction.nn == 0x07
    is_0x0A = instruction.nn == 0x0A
    is_0x15 = instruction.nn == 0x15
    is_0x18 = instruction.nn == 0x18
    is_0x1E = instruction.nn == 0x1E
    is_0x29 = instruction.nn == 0x29
    is_0x33 = instruction.nn == 0x33
    is_0x55 = instruction.nn == 0x55
    is_0x65 = instruction.nn == 0x65

    switch_index = (
        is_0x07 * 0 +
        is_0x0A * 1 +
        is_0x15 * 2 +
        is_0x18 * 3 +
        is_0x1E * 4 +
        is_0x29 * 5 +
        is_0x33 * 6 +
        is_0x55 * 7 +
        is_0x65 * 8 +
        (~(is_0x07 | is_0x0A | is_0x15 | is_0x18 | is_0x1E | is_0x29 | is_0x33 | is_0x55 | is_0x65)) * 9
    )

    return jax.lax.switch(
        switch_index,
        [
            execute_get_delay_timer,
            execute_wait_for_key,
            execute_set_delay_timer,
            execute_set_sound_timer,
            execute_add_to_index,
            execute_font_character,
            execute_bcd_conversion,
            execute_store_registers,
            execute_load_registers,
            no_op,
        ],
        state, instruction
    )