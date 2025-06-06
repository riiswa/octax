"""Tests for ALU operations (8xxx)."""

import pytest
from octax import execute


class TestBasicALU:
    """Test basic ALU operations."""

    def test_alu_set_basic(self, fresh_state):
        """8XY0 - Set VX = VY."""
        state = fresh_state
        state = state.replace(V=state.V.at[1].set(0x42))
        state = state.replace(V=state.V.at[2].set(0x99))

        state = execute(state, 0x8120)  # V1 = V2

        assert state.V[1] == 0x99
        assert state.V[2] == 0x99

    def test_alu_or_basic(self, fresh_state):
        """8XY1 - OR operation."""
        state = fresh_state
        state = state.replace(V=state.V.at[1].set(0xF0))
        state = state.replace(V=state.V.at[2].set(0x0F))

        state = execute(state, 0x8121)  # V1 |= V2

        assert state.V[1] == 0xFF
        assert state.V[15] == 0

    def test_alu_and_basic(self, fresh_state):
        """8XY2 - AND operation."""
        state = fresh_state
        state = state.replace(V=state.V.at[1].set(0xF0))
        state = state.replace(V=state.V.at[2].set(0xF1))

        state = execute(state, 0x8122)  # V1 &= V2

        assert state.V[1] == 0xF0
        assert state.V[15] == 0

    def test_alu_xor_basic(self, fresh_state):
        """8XY3 - XOR operation."""
        state = fresh_state
        state = state.replace(V=state.V.at[1].set(0xFF))
        state = state.replace(V=state.V.at[2].set(0xF0))

        state = execute(state, 0x8123)  # V1 ^= V2

        assert state.V[1] == 0x0F
        assert state.V[15] == 0

    def test_alu_xor_same(self, fresh_state):
        """8XY3 - XOR with same value should be 0."""
        state = fresh_state
        state = state.replace(V=state.V.at[3].set(0xAA))
        state = state.replace(V=state.V.at[4].set(0xAA))

        state = execute(state, 0x8343)  # V3 ^= V4

        assert state.V[3] == 0x00
        assert state.V[15] == 0


class TestALUArithmetic:
    """Test arithmetic ALU operations."""

    def test_alu_add_no_carry(self, fresh_state):
        """8XY4 - Add without carry."""
        state = fresh_state
        state = state.replace(V=state.V.at[1].set(0x10))
        state = state.replace(V=state.V.at[2].set(0x20))

        state = execute(state, 0x8124)  # V1 += V2

        assert state.V[1] == 0x30
        assert state.V[15] == 0

    def test_alu_add_with_carry(self, fresh_state):
        """8XY4 - Add with carry."""
        state = fresh_state
        state = state.replace(V=state.V.at[1].set(0xFF))
        state = state.replace(V=state.V.at[2].set(0x01))

        state = execute(state, 0x8124)  # V1 += V2

        assert state.V[1] == 0x00  # 256 wraps to 0
        assert state.V[15] == 1  # Carry set

    def test_alu_sub_xy_no_borrow(self, fresh_state):
        """8XY5 - Subtract VX - VY, no borrow."""
        state = fresh_state
        state = state.replace(V=state.V.at[1].set(0x30))
        state = state.replace(V=state.V.at[2].set(0x10))

        state = execute(state, 0x8125)  # V1 -= V2

        assert state.V[1] == 0x20
        assert state.V[15] == 1  # No borrow (VX >= VY)

    def test_alu_sub_xy_with_borrow(self, fresh_state):
        """8XY5 - Subtract VX - VY, with borrow."""
        state = fresh_state
        state = state.replace(V=state.V.at[3].set(0x10))
        state = state.replace(V=state.V.at[4].set(0x30))

        state = execute(state, 0x8345)  # V3 -= V4

        assert state.V[3] == 0xE0  # 16 - 48 = -32 → 224
        assert state.V[15] == 0  # Borrow (VX < VY)

    def test_alu_sub_yx_no_borrow(self, fresh_state):
        """8XY7 - Subtract VY - VX, no borrow."""
        state = fresh_state
        state = state.replace(V=state.V.at[1].set(0x10))
        state = state.replace(V=state.V.at[2].set(0x30))

        state = execute(state, 0x8127)  # V1 = V2 - V1

        assert state.V[1] == 0x20  # 48 - 16 = 32
        assert state.V[15] == 1  # No borrow (VY >= VX)


class TestALUShifts:
    """Test shift operations with mode differences."""

    def test_shift_right_modern_even(self, modern_state):
        """8XY6 - Shift right, modern mode, even number."""
        state = modern_state.replace(V=modern_state.V.at[1].set(0x04))
        state = state.replace(V=state.V.at[2].set(0xFF))  # Should be ignored

        state = execute(state, 0x8126)  # V1 >>= 1

        assert state.V[1] == 0x02  # 4 >> 1 = 2
        assert state.V[15] == 0  # LSB was 0

    def test_shift_right_modern_odd(self, modern_state):
        """8XY6 - Shift right, modern mode, odd number."""
        state = modern_state.replace(V=modern_state.V.at[3].set(0x05))
        state = state.replace(V=state.V.at[4].set(0xFF))  # Should be ignored

        state = execute(state, 0x8346)  # V3 >>= 1

        assert state.V[3] == 0x02  # 5 >> 1 = 2
        assert state.V[15] == 1  # LSB was 1

    def test_shift_right_legacy(self, legacy_state):
        """8XY6 - Shift right, legacy mode."""
        state = legacy_state.replace(V=legacy_state.V.at[5].set(0x08))  # Should be ignored
        state = state.replace(V=state.V.at[6].set(0x03))  # 00000011

        state = execute(state, 0x8566)  # V5 = V6 >> 1

        assert state.V[5] == 0x01  # V6 (3) >> 1 = 1
        assert state.V[15] == 1  # LSB of V6 was 1

    def test_shift_left_modern_overflow(self, modern_state):
        """8XYE - Shift left, modern mode, with overflow."""
        state = modern_state.replace(V=modern_state.V.at[3].set(0x81))  # 10000001
        state = state.replace(V=state.V.at[4].set(0xFF))  # Should be ignored

        state = execute(state, 0x834E)  # V3 <<= 1

        assert state.V[3] == 0x02  # 129 << 1 = 258 → 2
        assert state.V[15] == 1  # MSB was 1

    def test_shift_mode_comparison(self):
        """Test that modern_mode correctly switches behaviors."""
        from chix8 import create_state

        # Modern: V1 = V1 >> 1 (ignores V2)
        state_modern = create_state().replace(modern_mode=True)
        state_modern = state_modern.replace(V=state_modern.V.at[1].set(0x08))  # V1 = 8
        state_modern = state_modern.replace(V=state_modern.V.at[2].set(0x03))  # V2 = 3
        state_modern = execute(state_modern, 0x8126)

        # Legacy: V1 = V2 >> 1 (uses V2)
        state_legacy = create_state().replace(modern_mode=False)
        state_legacy = state_legacy.replace(V=state_legacy.V.at[1].set(0x08))  # V1 = 8
        state_legacy = state_legacy.replace(V=state_legacy.V.at[2].set(0x03))  # V2 = 3
        state_legacy = execute(state_legacy, 0x8126)

        assert state_modern.V[1] == 0x04  # 8 >> 1 = 4 (used V1)
        assert state_legacy.V[1] == 0x01  # 3 >> 1 = 1 (used V2)


class TestALUEdgeCases:
    """Test edge cases and comprehensive scenarios."""

    def test_alu_undefined_operations(self, fresh_state):
        """Test undefined ALU operations."""
        undefined_ops = [0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xF]

        for op in undefined_ops:
            state = fresh_state
            state = state.replace(V=state.V.at[1].set(0x42))
            state = state.replace(V=state.V.at[2].set(0x99))

            instruction = 0x8120 | op
            state = execute(state, instruction)

            assert state.V[1] == 0x42, f"Undefined op {op:X} changed VX"
            assert state.V[15] == 0, f"Undefined op {op:X} set VF to non-zero"

    def test_alu_self_operations(self, fresh_state):
        """Test operations where VX and VY are the same register."""
        state = fresh_state.replace(V=fresh_state.V.at[5].set(0xAA))

        # V5 ^= V5 (should become 0)
        state = execute(state, 0x8553)
        assert state.V[5] == 0x00, "Self XOR should result in 0"

        # Reset and test self ADD
        state = state.replace(V=state.V.at[5].set(0x80))
        state = execute(state, 0x8554)  # V5 += V5
        assert state.V[5] == 0x00, "Self ADD should wrap on overflow"
        assert state.V[15] == 1, "Self ADD should set carry flag"

    def test_vf_register_operations(self, fresh_state):
        """Test that operations on VF work correctly."""
        state = fresh_state
        state = state.replace(V=state.V.at[15].set(0x42))  # VF = 0x42
        state = state.replace(V=state.V.at[1].set(0x10))

        state = execute(state, 0x81F4)  # V1 += VF
        assert state.V[1] == 0x52, "Addition with VF as source failed"
        assert state.V[15] == 0, "VF should be overwritten by operation result"