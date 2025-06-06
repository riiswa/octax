"""Tests for display operations (DXYN)."""

import pytest
from octax import execute
from conftest import setup_sprite_in_memory


class TestBasicSprites:
    """Test basic sprite drawing."""

    def test_basic_sprite_draw(self, fresh_state):
        """Test basic sprite drawing without collision."""
        state = fresh_state

        # Simple 2x2 box sprite
        sprite = [0xC0, 0xC0]  # 11000000, 11000000
        state = setup_sprite_in_memory(state, 0x300, sprite)

        # Set coordinates: V0=10, V1=5
        state = execute(state, 0x600A)  # V0 = 10
        state = execute(state, 0x6105)  # V1 = 5
        state = execute(state, 0xA300)  # I = 0x300

        # Draw sprite: D012 (draw at V0,V1 with height 2)
        state = execute(state, 0xD012)

        # Check pixels are drawn
        assert state.display[10, 5] == 1  # Top-left
        assert state.display[11, 5] == 1  # Top-right
        assert state.display[10, 6] == 1  # Bottom-left
        assert state.display[11, 6] == 1  # Bottom-right
        assert state.display[12, 5] == 0  # Outside sprite

        # No collision should occur
        assert state.V[15] == 0

    def test_collision_detection(self, fresh_state):
        """Test collision flag when sprite overlaps existing pixels."""
        state = fresh_state

        # Single pixel sprite
        sprite = [0x80]  # 10000000
        state = setup_sprite_in_memory(state, 0x400, sprite)

        # Set coordinates and I register
        state = execute(state, 0x6014)  # V0 = 20
        state = execute(state, 0x610A)  # V1 = 10
        state = execute(state, 0xA400)  # I = 0x400

        # Draw first time - no collision
        state = execute(state, 0xD011)  # Draw height 1
        assert state.display[20, 10] == 1
        assert state.V[15] == 0  # No collision

        # Draw again at same location - should collision
        state = execute(state, 0xD011)  # Draw again
        assert state.display[20, 10] == 0  # Pixel erased by XOR
        assert state.V[15] == 1  # Collision detected!

    def test_xor_behavior(self, fresh_state):
        """Test XOR behavior - drawing twice should erase."""
        state = fresh_state

        # Line sprite
        sprite = [0xF0]  # 11110000
        state = setup_sprite_in_memory(state, 0x500, sprite)

        state = execute(state, 0x6008)  # V0 = 8
        state = execute(state, 0x610F)  # V1 = 15
        state = execute(state, 0xA500)  # I = 0x500

        # Draw first time
        state = execute(state, 0xD011)
        assert state.display[8, 15] == 1
        assert state.display[9, 15] == 1
        assert state.display[10, 15] == 1
        assert state.display[11, 15] == 1
        assert state.V[15] == 0  # No collision first time

        # Draw second time - should erase
        state = execute(state, 0xD011)
        assert state.display[8, 15] == 0
        assert state.display[9, 15] == 0
        assert state.display[10, 15] == 0
        assert state.display[11, 15] == 0
        assert state.V[15] == 1  # Collision detected


class TestScreenBoundaries:
    """Test sprite clipping and wrapping."""

    def test_screen_boundaries(self, fresh_state):
        """Test sprites at screen edges."""
        state = fresh_state

        # 8x1 full width sprite
        sprite = [0xFF]  # 11111111
        state = setup_sprite_in_memory(state, 0x600, sprite)

        # Draw at right edge (x=60, width=8 → pixels 60-67, but screen is 0-63)
        state = execute(state, 0x603C)  # V0 = 60
        state = execute(state, 0x6100)  # V1 = 0
        state = execute(state, 0xA600)  # I = 0x600

        state = execute(state, 0xD011)

        # Should only draw pixels 60-63 (4 pixels visible)
        assert state.display[60, 0] == 1
        assert state.display[61, 0] == 1
        assert state.display[62, 0] == 1
        assert state.display[63, 0] == 1

    def test_bottom_edge_clipping(self, fresh_state):
        """Test sprite clipping at bottom edge."""
        state = fresh_state

        # 3-row sprite
        sprite = [0x80, 0x80, 0x80]  # Three pixels vertically
        state = setup_sprite_in_memory(state, 0x700, sprite)

        # Draw at bottom edge (y=30, height=3 → rows 30,31,32 but screen is 0-31)
        state = execute(state, 0x6000)  # V0 = 0
        state = execute(state, 0x611E)  # V1 = 30
        state = execute(state, 0xA700)  # I = 0x700

        state = execute(state, 0xD013)  # Draw height 3

        # Should only draw rows 30-31 (2 rows visible)
        assert state.display[0, 30] == 1
        assert state.display[0, 31] == 1

    def test_coordinate_wrapping(self, fresh_state):
        """Test coordinate wrapping with modulo."""
        state = fresh_state

        sprite = [0x80]  # Single pixel
        state = setup_sprite_in_memory(state, 0x800, sprite)

        # Set coordinates > screen size to test modulo
        state = execute(state, 0x6046)  # V0 = 70 (70 % 64 = 6)
        state = execute(state, 0x6125)  # V1 = 37 (37 % 32 = 5)
        state = execute(state, 0xA800)  # I = 0x800

        state = execute(state, 0xD011)

        # Should draw at (6, 5) due to modulo
        assert state.display[6, 5] == 1


class TestSpriteVariations:
    """Test different sprite configurations."""

    def test_different_sprite_heights(self, fresh_state):
        """Test sprites with different N values."""
        state = fresh_state

        # Multi-row sprite
        sprite = [0x80, 0x40, 0x20, 0x10, 0x08]  # Diagonal line
        state = setup_sprite_in_memory(state, 0x900, sprite)

        state = execute(state, 0x600A)  # V0 = 10
        state = execute(state, 0x6108)  # V1 = 8
        state = execute(state, 0xA900)  # I = 0x900

        # Draw only first 3 rows (N=3)
        state = execute(state, 0xD013)

        # Check only first 3 pixels of diagonal
        assert state.display[10, 8] == 1  # Row 0: 0x80 → bit 7
        assert state.display[11, 9] == 1  # Row 1: 0x40 → bit 6
        assert state.display[12, 10] == 1  # Row 2: 0x20 → bit 5
        assert state.display[13, 11] == 0  # Row 3: not drawn (N=3)

    def test_vf_register_preservation(self, fresh_state):
        """Test that VF is properly set/cleared."""
        state = fresh_state

        sprite = [0x80]
        state = setup_sprite_in_memory(state, 0xB00, sprite)

        # Set VF to 1 initially
        state = execute(state, 0x6F01)  # VF = 1

        # Draw sprite with no collision
        state = execute(state, 0x6005)  # V0 = 5
        state = execute(state, 0x6105)  # V1 = 5
        state = execute(state, 0xAB00)  # I = 0xB00
        state = execute(state, 0xD011)

        # VF should be cleared to 0 (no collision)
        assert state.V[15] == 0