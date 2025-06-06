"""CHIP-8 stack operations."""

import jax.numpy as jnp
from octax.constants import ADDRESS_MASK
from octax.state import StackState


def push(stack: StackState, address: jnp.ndarray) -> StackState:
    """Push address onto stack."""
    masked_address = address & ADDRESS_MASK
    new_data = stack.data.at[stack.pointer].set(masked_address)
    return stack.replace(data=new_data, pointer=stack.pointer + 1)


def pop(stack: StackState) -> tuple[StackState, jnp.ndarray]:
    """Pop address from stack."""
    new_pointer = stack.pointer - 1
    popped_address = stack.data[new_pointer]
    new_data = stack.data.at[new_pointer].set(0)
    return stack.replace(data=new_data, pointer=new_pointer), popped_address