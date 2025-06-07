"""CHIP-8 ALU operations (8xxx)."""

import jax
import jax.lax
import jax.numpy as jnp
from octax.state import EmulatorState
from octax.decode import DecodedInstruction


def alu_set(vx: int, vy: int) -> tuple[int, int]:
    """8XY0 - Set: VX = VY."""
    return vy, jnp.zeros((), dtype=jnp.uint8)


def alu_or(vx: int, vy: int) -> tuple[int, int]:
    """8XY1 - Binary OR: VX |= VY."""
    return vx | vy, jnp.zeros((), dtype=jnp.uint8)


def alu_and(vx: int, vy: int) -> tuple[int, int]:
    """8XY2 - Binary AND: VX &= VY."""
    return vx & vy, jnp.zeros((), dtype=jnp.uint8)


def alu_xor(vx: int, vy: int) -> tuple[int, int]:
    """8XY3 - Logical XOR: VX ^= VY."""
    return vx ^ vy, jnp.zeros((), dtype=jnp.uint8)


def alu_add(vx: int, vy: int) -> tuple[int, int]:
    """8XY4 - Add: VX += VY, set carry flag."""
    result = jnp.astype(vx, jnp.int32) + vy
    carry = jnp.astype(result > 255, jnp.uint8)
    return jnp.astype(result, jnp.uint8) & 0xFF, carry


def alu_sub_xy(vx: int, vy: int) -> tuple[int, int]:
    """8XY5 - Subtract: VX -= VY, set borrow flag."""
    borrow_flag = jnp.astype(vx >= vy, jnp.uint8)
    result = (vx - vy) & 0xFF
    return result, borrow_flag


def alu_shift_right(vx: int, vy: int) -> tuple[int, int]:
    """8XY6 - Shift right: VX >>= 1."""
    shifted_bit = vx & 1
    result = vx >> 1
    return result, shifted_bit


def alu_shift_left(vx: int, vy: int) -> tuple[int, int]:
    """8XYE - Shift left: VX <<= 1."""
    shifted_bit = (vx & 0x80) >> 7
    result = (vx << 1) & 0xFF
    return result, shifted_bit


def alu_sub_yx(vx: int, vy: int) -> tuple[int, int]:
    """8XY7 - Subtract: VX = VY - VX, set borrow flag."""
    borrow_flag = jnp.astype(vy >= vx, jnp.uint8)
    result = (vy - vx) & 0xFF
    return result, borrow_flag


def alu_undefined(vx: int, vy: int) -> tuple[int, int]:
    """Undefined ALU operation."""
    return vx, jnp.zeros((), dtype=jnp.uint8)


def execute_alu_operation(state: EmulatorState, instruction: DecodedInstruction) -> EmulatorState:
    """8XYN - ALU operations dispatcher."""
    vx = state.V[instruction.x]
    vy = state.V[instruction.y]

    def _alu_shift_left(vx: int, vy: int) -> tuple[int, int]:
        if not state.modern_mode:
            vx = vy
        return alu_shift_left(vx, vy)

    def _alu_shift_right(vx: int, vy: int) -> tuple[int, int]:
        if not state.modern_mode:
            vx = vy
        return alu_shift_right(vx, vy)

    # Use bitmask to handle undefined operations more efficiently
    valid_ops = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0], dtype=bool)  # Mark valid operations

    result, vf = jax.lax.cond(
        valid_ops[instruction.n],
        lambda: jax.lax.switch(
            # Map only valid operations: 0,1,2,3,4,5,6,7,14 -> 0,1,2,3,4,5,6,7,8
            jnp.where(instruction.n == 14, 8, instruction.n),
            [alu_set, alu_or, alu_and, alu_xor, alu_add,
             alu_sub_xy, _alu_shift_right, alu_sub_yx, _alu_shift_left],
            vx, vy
        ),
        lambda: (vx, jnp.zeros((), dtype=jnp.uint8))  # Single undefined handler
    )

    new_V = state.V.at[instruction.x].set(result)
    new_V = new_V.at[15].set(vf)
    return state.replace(V=new_V)