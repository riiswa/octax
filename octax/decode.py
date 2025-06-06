"""CHIP-8 instruction decoding."""

from chex import dataclass


@dataclass(frozen=True)
class DecodedInstruction:
    """Decoded CHIP-8 instruction with extracted operands."""
    raw: int
    opcode: int  # First nibble
    x: int       # Second nibble (VX register)
    y: int       # Third nibble (VY register)
    n: int       # Fourth nibble (4-bit immediate)
    nn: int      # Last byte (8-bit immediate)
    nnn: int     # Last 12 bits (12-bit address)


def decode(instruction: int) -> DecodedInstruction:
    """Decode 16-bit instruction into components."""
    return DecodedInstruction(
        raw=instruction,
        opcode=(instruction & 0xF000) >> 12,
        x=(instruction & 0x0F00) >> 8,
        y=(instruction & 0x00F0) >> 4,
        n=instruction & 0x000F,
        nn=instruction & 0x00FF,
        nnn=instruction & 0x0FFF
    )