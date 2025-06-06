"""Tests for memory and register operations."""

import pytest
from octax import execute


class TestBasicMemory:
    """Test basic memory operations."""

    def test_set_basic(self, fresh_state):
        """6XNN - Set VX = NN."""
        state = execute(fresh_state, 0x600A)  # V0 = 0xA
        assert state.V[0] == 0xA

    def test_add_basic(self, fresh_state):
        """7XNN - Add NN to VX."""
        state = fresh_state.replace(V=fresh_state.V.at[1].set(0x10))
        state = execute(state, 0x7105)  # V1 += 5
        assert state.V[1] == 0x15


class TestIndexRegister:
    """Test I register operations."""

    def test_set_index_basic(self, fresh_state):
        """ANNN - Set I register to NNN."""
        state = execute(fresh_state, 0xA123)  # I = 0x123
        assert state.I == 0x123

    def test_set_index_zero(self, fresh_state):
        """ANNN - Set I register to zero."""
        state = execute(fresh_state, 0xA123)  # I = 0x123
        state = execute(state, 0xA000)  # I = 0x000
        assert state.I == 0x000

    def test_set_index_maximum(self, fresh_state):
        """ANNN - Set I register to maximum 12-bit value."""
        state = execute(fresh_state, 0xAFFF)  # I = 0xFFF
        assert state.I == 0xFFF

    def test_set_index_common_values(self, fresh_state):
        """ANNN - Test common memory addresses."""
        test_values = [0x200, 0x300, 0x500, 0x600, 0xA00, 0xEA0]

        for value in test_values:
            state = fresh_state
            instruction = 0xA000 | value

            state = execute(state, instruction)

            assert state.I == value, f"Failed to set I to 0x{value:03X}"

    def test_set_index_multiple_operations(self, fresh_state):
        """ANNN - Test multiple consecutive I register sets."""
        state = fresh_state

        state = execute(state, 0xA111)  # I = 0x111
        assert state.I == 0x111

        state = execute(state, 0xA222)  # I = 0x222
        assert state.I == 0x222

        state = execute(state, 0xA000)  # I = 0x000
        assert state.I == 0x000


class TestRandom:
    """Test random number generation."""

    def test_random_zero_mask(self, fresh_state):
        """CXNN - Random AND with 0x00 should always be 0."""
        state = execute(fresh_state, 0xC000)  # V0 = random & 0x00
        assert state.V[0] == 0

    def test_random_full_mask(self, fresh_state):
        """CXNN - Random AND with 0xFF should preserve full random value."""
        state = execute(fresh_state, 0xC1FF)  # V1 = random & 0xFF
        assert 0 <= state.V[1] <= 255

    def test_random_bit_mask(self, fresh_state):
        """CXNN - Random AND with specific mask."""
        state = execute(fresh_state, 0xC20F)  # V2 = random & 0x0F
        assert 0 <= state.V[2] <= 15

    def test_random_different_registers(self, fresh_state):
        """CXNN - Test random works with different registers."""
        state = fresh_state

        state = execute(state, 0xC3FF)  # V3 = random & 0xFF
        state = execute(state, 0xC4FF)  # V4 = random & 0xFF
        state = execute(state, 0xC5FF)  # V5 = random & 0xFF

        assert 0 <= state.V[3] <= 255
        assert 0 <= state.V[4] <= 255
        assert 0 <= state.V[5] <= 255

    def test_random_preserves_state(self, fresh_state):
        """CXNN - Verify other state is preserved."""
        state = fresh_state

        # Set up state
        state = execute(state, 0x6142)  # V1 = 0x42
        state = execute(state, 0x6299)  # V2 = 0x99
        state = execute(state, 0xA300)  # I = 0x300

        # Store original values
        original_V1 = state.V[1]
        original_V2 = state.V[2]
        original_I = state.I

        state = execute(state, 0xC0FF)  # V0 = random & 0xFF

        # Check other state preserved
        assert state.V[1] == original_V1
        assert state.V[2] == original_V2
        assert state.I == original_I

    def test_random_mask_patterns(self, fresh_state):
        """CXNN - Test various mask patterns."""
        state = fresh_state

        masks_and_max = [
            (0x01, 1),  # Only bit 0: result 0-1
            (0x03, 3),  # Bits 0-1: result 0-3
            (0x07, 7),  # Bits 0-2: result 0-7
            (0x80, 128),  # Only bit 7: result 0 or 128
        ]

        for i, (mask, max_val) in enumerate(masks_and_max):
            reg = i + 6  # Use registers V6, V7, V8, V9
            instruction = 0xC000 | (reg << 8) | mask

            state = execute(state, instruction)

            assert 0 <= state.V[reg] <= max_val, f"Mask 0x{mask:02X} failed"