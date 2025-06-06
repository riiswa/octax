"""Tests for miscellaneous instructions (Fxxx)."""

import pytest
from octax import execute


class TestTimers:
    """Test timer-related instructions."""

    def test_misc_timer_instructions(self, fresh_state):
        """Test timer set and get operations."""
        state = fresh_state

        # Test FX15: Set delay timer
        state = execute(state, 0x6030)  # V0 = 48
        state = execute(state, 0xF015)  # Set delay timer to V0
        assert state.delay_timer == 48

        # Test FX18: Set sound timer
        state = execute(state, 0x6120)  # V1 = 32
        state = execute(state, 0xF118)  # Set sound timer to V1
        assert state.sound_timer == 32

        # Test FX07: Get delay timer
        state = execute(state, 0xF207)  # V2 = delay timer
        assert state.V[2] == 48


class TestBCD:
    """Test BCD conversion."""

    def test_misc_bcd_conversion(self, fresh_state):
        """Test BCD conversion with 156."""
        state = fresh_state

        # Test with 156 (0x9C)
        state = execute(state, 0x609C)  # V0 = 156
        state = execute(state, 0xA300)  # I = 0x300
        state = execute(state, 0xF033)  # BCD conversion

        assert state.memory[0x300] == 1  # Hundreds
        assert state.memory[0x301] == 5  # Tens
        assert state.memory[0x302] == 6  # Ones

    def test_bcd_edge_cases(self, fresh_state):
        """Test BCD with edge cases."""
        state = fresh_state

        # Test with 0
        state = execute(state, 0x6000)  # V0 = 0
        state = execute(state, 0xA400)  # I = 0x400
        state = execute(state, 0xF033)  # BCD conversion

        assert state.memory[0x400] == 0  # Hundreds
        assert state.memory[0x401] == 0  # Tens
        assert state.memory[0x402] == 0  # Ones

        # Test with 255 (max)
        state = execute(state, 0x60FF)  # V0 = 255
        state = execute(state, 0xA500)  # I = 0x500
        state = execute(state, 0xF033)  # BCD conversion

        assert state.memory[0x500] == 2  # Hundreds
        assert state.memory[0x501] == 5  # Tens
        assert state.memory[0x502] == 5  # Ones


class TestFont:
    """Test font character addressing."""

    def test_misc_font_character(self, fresh_state):
        """Test font character addressing."""
        state = fresh_state

        # Test character 'A' (0xA)
        state = execute(state, 0x600A)  # V0 = 0xA
        state = execute(state, 0xF029)  # I = font address for A

        expected_address = 0x50 + (0xA * 5)  # 0x50 + 50 = 0x82
        assert state.I == expected_address

    def test_font_all_characters(self, fresh_state):
        """Test font addressing for all hex digits."""
        state = fresh_state

        for digit in range(16):
            state = execute(state, 0x6000 | digit)  # V0 = digit
            state = execute(state, 0xF029)  # I = font address

            expected = 0x50 + (digit * 5)
            assert state.I == expected, f"Font address wrong for digit {digit:X}"


class TestMemoryOperations:
    """Test store/load register operations."""

    def test_store_load_modern_mode(self, modern_state):
        """Test store/load with modern_mode (I doesn't change)."""
        state = modern_state

        # Set up test data
        state = execute(state, 0x6001)  # V0 = 1
        state = execute(state, 0x6102)  # V1 = 2
        state = execute(state, 0x6203)  # V2 = 3
        state = execute(state, 0xA300)  # I = 0x300

        original_i = state.I

        # Store registers
        state = execute(state, 0xF255)  # Store V0-V2
        assert state.I == original_i  # I unchanged in modern mode

        # Clear registers
        state = execute(state, 0x6000)  # V0 = 0
        state = execute(state, 0x6100)  # V1 = 0
        state = execute(state, 0x6200)  # V2 = 0

        # Load back
        state = execute(state, 0xF265)  # Load V0-V2
        assert state.V[0] == 1
        assert state.V[1] == 2
        assert state.V[2] == 3
        assert state.I == original_i  # I still unchanged

    def test_store_load_legacy_mode(self, legacy_state):
        """Test store/load with legacy_mode (I increments)."""
        state = legacy_state

        # Set up test data
        state = execute(state, 0x6001)  # V0 = 1
        state = execute(state, 0x6102)  # V1 = 2
        state = execute(state, 0xA400)  # I = 0x400

        # Store registers
        state = execute(state, 0xF155)  # Store V0-V1
        assert state.I == 0x400 + 2  # I incremented by X+1

        # Reset I for load test
        state = execute(state, 0xA400)  # I = 0x400

        # Clear registers
        state = execute(state, 0x6000)  # V0 = 0
        state = execute(state, 0x6100)  # V1 = 0

        # Load back
        state = execute(state, 0xF165)  # Load V0-V1
        assert state.V[0] == 1
        assert state.V[1] == 2
        assert state.I == 0x400 + 2  # I incremented again


class TestKeypad:
    """Test keypad operations."""

    def test_skip_if_key_pressed(self, fresh_state):
        """Test EX9E - Skip if key pressed."""
        state = fresh_state

        # Set V0 = 5 and press key 5
        state = execute(state, 0x6005)  # V0 = 5
        state = state.replace(keypad=state.keypad.at[5].set(True))
        initial_pc = state.pc

        state = execute(state, 0xE09E)  # Skip if key V0 pressed
        assert state.pc == initial_pc + 2  # Should skip

    def test_skip_if_key_not_pressed(self, fresh_state):
        """Test EXA1 - Skip if key not pressed."""
        state = fresh_state

        # Set V0 = 5 and don't press key 5
        state = execute(state, 0x6005)  # V0 = 5
        initial_pc = state.pc

        state = execute(state, 0xE0A1)  # Skip if key V0 not pressed
        assert state.pc == initial_pc + 2  # Should skip

    def test_wait_for_key_blocking(self, fresh_state):
        """Test FX0A - Wait for key (blocking behavior)."""
        state = fresh_state
        initial_pc = state.pc

        # Execute wait instruction with no key pressed
        state = execute(state, 0xF00A)  # Wait for key → V0

        # PC should be decremented (instruction repeats)
        assert state.pc == initial_pc - 2

    def test_wait_for_key_release(self, fresh_state):
        """Test FX0A - Wait for key (key press behavior)."""
        state = fresh_state

        # Press key 7
        state = state.replace(keypad=state.keypad.at[7].set(True))
        initial_pc = state.pc

        state = execute(state, 0xF00A)  # Wait for key → V0

        # Should store pressed key and continue
        assert state.V[0] == 7
        assert state.pc == initial_pc  # PC not decremented


class TestMiscInstructionDispatch:
    """Test misc instruction dispatch logic."""

    def test_add_to_index(self, fresh_state):
        """Test FX1E - Add VX to I register."""
        state = fresh_state

        state = execute(state, 0x6010)  # V0 = 0x10
        state = execute(state, 0xA300)  # I = 0x300
        state = execute(state, 0xF01E)  # I += V0

        assert state.I == 0x310
        assert state.V[15] == 0  # No overflow

    def test_add_to_index_overflow(self, fresh_state):
        """Test FX1E with overflow."""
        state = fresh_state

        state = execute(state, 0x60FF)  # V0 = 0xFF
        state = execute(state, 0xAF80)  # I = 0xF80
        state = execute(state, 0xF01E)  # I += V0

        assert state.I == 0x07F  # Wrapped to 12-bit
        assert state.V[15] == 1  # Overflow flag set