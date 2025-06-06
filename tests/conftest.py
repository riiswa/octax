"""Test configuration and fixtures for CHIP-8 emulator tests."""

import pytest
import jax.numpy as jnp
from octax import create_state


@pytest.fixture
def fresh_state():
    """Provide a fresh emulator state for each test."""
    return create_state()


@pytest.fixture
def modern_state():
    """Provide a fresh state in modern mode."""
    return create_state().replace(modern_mode=True)


@pytest.fixture
def legacy_state():
    """Provide a fresh state in legacy mode."""
    return create_state().replace(modern_mode=False)


def setup_sprite_in_memory(state, address, sprite_bytes):
    """Helper to put sprite data in memory."""
    return state.replace(
        memory=state.memory.at[address:address+len(sprite_bytes)].set(
            jnp.array(sprite_bytes, dtype=jnp.uint8)
        )
    )