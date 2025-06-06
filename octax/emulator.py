"""Main CHIP-8 emulator execution engine."""

import jax
import jax.lax
import jax.numpy as jnp
from octax.state import EmulatorState
from octax.decode import decode
from octax.constants import PROGRAM_START
from octax.instructions.system import execute_system_instruction
from octax.instructions.control_flow import (
    execute_jump, execute_call, execute_skip_if_equal_immediate,
    execute_skip_if_not_equal_immediate, execute_skip_if_equal_register,
    execute_skip_if_not_equal_register, execute_jump_with_offset_modern,
    execute_jump_with_offset_legacy, execute_skip_if_key
)
from octax.instructions.alu import execute_alu_operation
from octax.instructions.memory import execute_set, execute_add, execute_set_index, execute_random
from octax.instructions.display import execute_display
from octax.instructions.misc import execute_misc_instruction


def execute(state: EmulatorState, instruction: int) -> EmulatorState:
    """Execute single CHIP-8 instruction."""
    decoded_instruction = decode(instruction)

    return jax.lax.switch(
        decoded_instruction.opcode,
        [
            execute_system_instruction,
            execute_jump,
            execute_call,
            execute_skip_if_equal_immediate,
            execute_skip_if_not_equal_immediate,
            execute_skip_if_equal_register,
            execute_set,
            execute_add,
            execute_alu_operation,
            execute_skip_if_not_equal_register,
            execute_set_index,
            execute_jump_with_offset_modern if state.modern_mode else execute_jump_with_offset_legacy,
            execute_random,
            execute_display,
            execute_skip_if_key,
            execute_misc_instruction,
        ],
        state, decoded_instruction
    )

def _pack_u16(high: jnp.uint8, low: jnp.uint8) -> jnp.uint16:
    """Pack two bytes into uint16."""
    return (high.astype(jnp.uint16) << 8) | low.astype(jnp.uint16)


def _unpack_u16(value: jnp.uint16) -> tuple[jnp.uint8, jnp.uint8]:
    """Unpack uint16 into two bytes."""
    return (value >> 8).astype(jnp.uint8), (value & 0xFF).astype(jnp.uint8)


def fetch(state: EmulatorState) -> tuple[EmulatorState, jnp.uint16]:
    """Fetch next instruction from memory."""
    instruction = _pack_u16(state.memory[state.pc], state.memory[state.pc + 1])
    return state.replace(pc=state.pc + 2), instruction


def load_rom(state: EmulatorState, filename: str) -> EmulatorState:
    """Load ROM data into CHIP-8 memory starting at 0x200."""
    with open(filename, 'rb') as f:
        rom_data = f.read()
    rom_array = jnp.array(list(rom_data), dtype=jnp.uint8)
    new_memory = state.memory.at[PROGRAM_START:PROGRAM_START + len(rom_data)].set(rom_array)
    return state.replace(memory=new_memory)