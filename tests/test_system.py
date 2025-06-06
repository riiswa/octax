"""Tests for system instructions (0xxx)."""

import jax.numpy as jnp
from octax import execute


def test_execute_clear_screen(fresh_state):
    """Test 00E0 - Clear display."""
    state = fresh_state.replace(display=fresh_state.display.at[0, 0].set(1))

    state = execute(state, 0x00E0)

    assert jnp.sum(state.display) == 0


def test_execute_call_and_return(fresh_state):
    """Test 2NNN (call) and 00EE (return) together."""
    state = fresh_state
    initial_pc = state.pc

    # Call subroutine
    state = execute(state, 0x2300)  # Call 0x300
    assert state.pc == 0x300
    assert state.stack.data[state.stack.pointer - 1] == initial_pc

    # Return from subroutine
    state = execute(state, 0x00EE)  # Return
    assert state.pc == initial_pc


def test_execute_return_after_call(fresh_state):
    """Test return restores correct address."""
    state = fresh_state

    # Call then return
    state = execute(state, 0x2400)  # Call 0x400
    call_pc = state.pc
    state = execute(state, 0x00EE)  # Return

    assert state.pc != call_pc  # No longer at called address