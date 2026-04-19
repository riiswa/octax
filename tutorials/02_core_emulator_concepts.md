# Core Emulator Concepts

This tutorial covers the CHIP-8 emulator implementation in Octax. Understanding these concepts helps with troubleshooting, performance optimization, and extending functionality.

## The CHIP-8 Architecture

CHIP-8 isn't a real computer—it's a virtual machine specification from the 1970s designed to make game programming easier. Think of it as the "assembly language" of simple games. When you run a CHIP-8 game, you're actually running a tiny program written in CHIP-8 instructions on Octax's virtual processor.

The CHIP-8 system has several key components:
- **Memory**: 4KB of RAM where both the program and data live
- **Registers**: 16 general-purpose registers (V0-VF) for temporary storage
- **Stack**: For remembering where to return from subroutines
- **Timers**: For controlling game speed and sound
- **Display**: A 64x32 pixel monochrome screen

## The Emulator State

Everything in Octax revolves around the `EmulatorState`. This is a snapshot of the entire CHIP-8 system at any moment:

```python
import octax

# Create a fresh CHIP-8 system
state = octax.create_state()

print(f"Program counter starts at: 0x{state.pc:03X}")
print(f"Memory size: {len(state.memory)} bytes")
print(f"Number of registers: {len(state.v_registers)}")
print(f"Display dimensions: {state.display.shape}")
```

This design uses immutable state updates. Executing an instruction creates a new state rather than modifying the existing one, making it compatible with JAX's functional programming model.

## Loading and Running Programs

Every CHIP-8 game is stored as a ROM file—essentially a sequence of bytes that represent instructions and data. Let's see how to load and run a simple program:

```python
import octax
import jax.numpy as jnp

# Start with a fresh state
state = octax.create_state()

# Load a ROM (let's use a simple one)
with open("roms/Pong (1 player).ch8", "rb") as f:
    rom_data = jnp.array(list(f.read()), dtype=jnp.uint8)

# Load the ROM into memory
state = octax.load_rom(state, rom_data)

print(f"ROM loaded! First few bytes: {state.memory[0x200:0x210]}")
print(f"Program counter: 0x{state.pc:03X}")
```

CHIP-8 programs always load at address 0x200 (512 in decimal). The first 512 bytes are reserved for the interpreter and font data. This is why `state.pc` starts at 0x200—it's pointing to the first instruction of your program.

## The Execution Cycle

The heart of any emulator is the execution cycle: fetch, decode, execute, repeat. Let's break this down:

```python
# The basic emulation loop
def emulate_steps(state, num_steps):
    """Run the emulator for a specific number of steps"""
    for _ in range(num_steps):
        # 1. Fetch: Get the instruction at the current program counter
        state, instruction_word = octax.fetch(state)
        
        # 2. Decode: Figure out what this instruction means
        decoded = octax.decode(instruction_word)
        
        # 3. Execute: Carry out the instruction
        state = octax.execute(state, decoded)
        
        print(f"PC: 0x{state.pc:03X}, Instruction: 0x{instruction_word:04X}, "
              f"Opcode: {decoded.opcode}")
    
    return state

# Run a few steps to see what happens
state = emulate_steps(state, 5)
```

This three-step cycle is fundamental to computer emulation. The fetch step reads the instruction from memory and advances the program counter. The decode step interprets the raw bytes as a specific operation. The execute step actually performs that operation, potentially changing registers, memory, or display.

## Understanding Instructions

CHIP-8 has 35 different instruction types, each with its own behavior. Let's explore a few common ones:

```python
# Let's look at what instructions do
def analyze_instruction(instruction_word):
    decoded = octax.decode(instruction_word)
    
    if decoded.opcode == 0x6:  # Load immediate
        print(f"LD V{decoded.x:X}, 0x{decoded.nn:02X} - Load {decoded.nn} into register V{decoded.x:X}")
    elif decoded.opcode == 0x8 and decoded.n == 0x0:  # Copy register
        print(f"LD V{decoded.x:X}, V{decoded.y:X} - Copy V{decoded.y:X} to V{decoded.x:X}")
    elif decoded.opcode == 0x8 and decoded.n == 0x4:  # Add registers
        print(f"ADD V{decoded.x:X}, V{decoded.y:X} - Add V{decoded.y:X} to V{decoded.x:X}")
    elif decoded.opcode == 0xA:  # Set index register
        print(f"LD I, 0x{decoded.nnn:03X} - Set index register to 0x{decoded.nnn:03X}")
    elif decoded.opcode == 0xD:  # Draw sprite
        print(f"DRW V{decoded.x:X}, V{decoded.y:X}, {decoded.n} - Draw sprite at (V{decoded.x:X}, V{decoded.y:X})")
    else:
        print(f"Unknown or complex instruction: 0x{instruction_word:04X}")

# Analyze some common instructions
analyze_instruction(0x6A42)  # Load 0x42 into VA
analyze_instruction(0x8AB0)  # Copy VB to VA  
analyze_instruction(0x8AB4)  # Add VB to VA
analyze_instruction(0xA123)  # Set I to 0x123
analyze_instruction(0xDAB5)  # Draw 5-byte sprite at (VA, VB)
```

The instruction format follows consistent patterns:
- `6XNN`: Load immediate value NN into register VX
- `8XY0`: Copy register VY to register VX
- `ANNN`: Set the index register I to address NNN
- `DXYN`: Draw N-byte sprite from memory[I] at coordinates (VX, VY)

## Working with Registers and Memory

The CHIP-8 has 16 8-bit registers (V0 through VF) and one 16-bit index register (I). Register VF is special—it's used for flags like carry bits and collision detection:

```python
# Let's manipulate some registers
state = octax.create_state()

# Simulate loading values into registers
# In real CHIP-8, this would be done with 6XNN instructions
state = state.replace(v_registers=state.v_registers.at[0].set(100))  # V0 = 100
state = state.replace(v_registers=state.v_registers.at[1].set(50))   # V1 = 50

print(f"V0: {state.v_registers[0]}, V1: {state.v_registers[1]}")

# The index register is used for memory operations
state = state.replace(index_register=0x300)
print(f"Index register I: 0x{state.index_register:03X}")

# Memory can be read directly
print(f"Memory at I: {state.memory[state.index_register]}")
```

## The Display System

One of the most interesting aspects of CHIP-8 is its display system. Instead of setting individual pixels, you draw sprites using XOR operations:

```python
# Let's understand the display
state = octax.create_state()
print(f"Display shape: {state.display.shape}")
print(f"Display is all zeros initially: {jnp.all(state.display == 0)}")

# Drawing is done with XOR - this means drawing the same sprite twice erases it
# This was used for simple animation and collision detection

# The display buffer is just a 2D array of 0s and 1s
print(f"Display type: {state.display.dtype}")
```

The XOR drawing system enables sprite animation (draw, move, erase, redraw) and collision detection (XOR turning a pixel off indicates collision).

## Timers and Game Flow

CHIP-8 has two timers that count down at 60 Hz:

```python
# Fresh state has timers at 0
state = octax.create_state()
print(f"Delay timer: {state.delay_timer}")
print(f"Sound timer: {state.sound_timer}")

# Games set these timers to control timing
# For example, setting delay_timer = 60 creates a 1-second delay
state = state.replace(delay_timer=60, sound_timer=30)
print(f"Timers set - Delay: {state.delay_timer}, Sound: {state.sound_timer}")
```

The delay timer is used for game timing (like waiting between moves in Tetris), while the sound timer controls a simple beep sound. In Octax, these timers are automatically decremented as part of the environment step.

## Font Data and Text Rendering

CHIP-8 includes a built-in font for hexadecimal digits (0-F). Let's see how this works:

```python
from octax.constants import FONT_START

# The font data is loaded at startup
state = octax.create_state()

# Each character is 5 bytes tall and 4 pixels wide
# Let's look at the digit '0'
font_0 = state.memory[FONT_START:FONT_START+5]
print(f"Font data for '0': {font_0}")

# Convert to binary to see the pattern
for i, byte in enumerate(font_0):
    binary = format(byte, '08b')[:4]  # Only 4 bits are used
    print(f"Row {i}: {binary.replace('0', ' ').replace('1', '█')}")
```

This font system allows games to display scores and text. The index register is set to point to the desired character, then a draw instruction renders it to the screen.

## Putting It All Together

Now let's create a simple program that demonstrates these concepts:

```python
def step_by_step_execution(state, num_steps=10):
    """Execute instructions one by one with detailed output"""
    for step in range(num_steps):
        print(f"\n--- Step {step + 1} ---")
        print(f"PC: 0x{state.pc:03X}")
        
        # Check if we can fetch an instruction
        if state.pc >= len(state.memory) - 1:
            print("End of memory reached!")
            break
            
        # Fetch
        old_pc = state.pc
        state, instruction_word = octax.fetch(state)
        print(f"Fetched: 0x{instruction_word:04X} (PC: 0x{old_pc:03X} -> 0x{state.pc:03X})")
        
        # Decode
        decoded = octax.decode(instruction_word)
        print(f"Decoded - Opcode: 0x{decoded.opcode:X}, x: {decoded.x}, y: {decoded.y}, n: {decoded.n}")
        
        # Execute
        old_registers = state.v_registers.copy()
        state = octax.execute(state, decoded)
        
        # Show what changed
        register_changes = jnp.where(old_registers != state.v_registers)[0]
        if len(register_changes) > 0:
            for reg in register_changes:
                print(f"V{reg:X}: {old_registers[reg]} -> {state.v_registers[reg]}")
    
    return state

# Load a ROM and watch it execute
state = octax.create_state()
with open("roms/test_opcode.ch8", "rb") as f:
    rom_data = jnp.array(list(f.read()), dtype=jnp.uint8)
state = octax.load_rom(state, rom_data)

final_state = step_by_step_execution(state, 5)
```

## Modern vs Legacy Mode

Octax supports both modern and legacy CHIP-8 behavior. Some instructions behave differently between the original CHIP-8 interpreter and later implementations:
