"""
CHIP-8 Interactive Emulator using Pygame and Octax Engine

This module provides a complete pygame-based frontend for the octax CHIP-8 emulator.
It properly integrates with the octax architecture:

- Uses octax.state.EmulatorState with JAX arrays
- Uses octax.execute.fetch/execute functions
- Handles JAX array fields (pc, timers, I) correctly
- Supports both modern and legacy CHIP-8 modes
- Loads ROMs using octax.execute.load_rom

Key JAX Considerations:
- Convert JAX arrays to Python ints for display: int(state.pc)
- Timer updates use jnp.asarray() for proper JAX array creation
- Keypad state uses jnp.zeros() and .at[].set() operations
"""

import pygame
import jax
import jax.numpy as jnp
import time
from octax.state import EmulatorState, create_state
from octax import execute, fetch, load_rom
from octax.constants import SCREEN_WIDTH, SCREEN_HEIGHT

# CHIP-8 keypad mapping to keyboard
# CHIP-8 keypad:     Keyboard mapping:
#   1 2 3 C            1 2 3 4
#   4 5 6 D            Q W E R
#   7 8 9 E            A S D F
#   A 0 B F            Z X C V

KEY_MAP = {
    pygame.K_1: 0x1, pygame.K_2: 0x2, pygame.K_3: 0x3, pygame.K_4: 0xC,
    pygame.K_q: 0x4, pygame.K_w: 0x5, pygame.K_e: 0x6, pygame.K_r: 0xD,
    pygame.K_a: 0x7, pygame.K_s: 0x8, pygame.K_d: 0x9, pygame.K_f: 0xE,
    pygame.K_z: 0xA, pygame.K_x: 0x0, pygame.K_c: 0xB, pygame.K_v: 0xF
}


def run_interactive_emulator(initial_state: EmulatorState, rom_filename=None, scale=10, fps=60):
    """
    Run CHIP-8 emulator with pygame interactive loop using octax functions

    Args:
        initial_state: Initial EmulatorState
        rom_filename: Optional ROM file path to load
        scale: Display scale factor (10 = 640x320 window)
        fps: Target FPS for emulation
    """

    # Initialize pygame
    pygame.init()

    # Display settings - use constants from octax
    window_width = SCREEN_WIDTH * scale
    window_height = SCREEN_HEIGHT * scale

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("CHIP-8 Emulator - Octax")
    clock = pygame.time.Clock()

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)  # For active pixels

    # Load ROM if provided
    state = initial_state
    if rom_filename is not None:
        try:
            state = load_rom(state, rom_filename)
            print(f"Loaded ROM: {rom_filename}")
        except FileNotFoundError:
            print(f"ROM file not found: {rom_filename}")
        except Exception as e:
            print(f"Error loading ROM: {e}")

    # Initialize keypad state
    keypad_state = jnp.zeros(16, dtype=jnp.bool_)

    # Timing
    instruction_count = 0
    last_timer_update = time.time()

    running = True
    paused = False

    print("CHIP-8 Emulator Controls:")
    print("ESC - Quit")
    print("SPACE - Pause/Resume")
    print("R - Reset")
    print("Keypad mapping: 1234/QWER/ASDF/ZXCV")
    print("-" * 40)

    while running:
        dt = clock.tick(fps) / 1000.0  # Delta time in seconds

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
                elif event.key == pygame.K_r:
                    # Reset emulator
                    state = initial_state
                    if rom_filename is not None:
                        try:
                            state = load_rom(state, rom_filename)
                        except Exception as e:
                            print(f"Error reloading ROM: {e}")
                    keypad_state = jnp.zeros(16, dtype=jnp.bool_)
                    instruction_count = 0
                    print("Reset emulator")

                # Handle CHIP-8 keypad
                if event.key in KEY_MAP:
                    key_index = KEY_MAP[event.key]
                    keypad_state = keypad_state.at[key_index].set(True)

            elif event.type == pygame.KEYUP:
                # Handle CHIP-8 keypad release
                if event.key in KEY_MAP:
                    key_index = KEY_MAP[event.key]
                    keypad_state = keypad_state.at[key_index].set(False)

        # Update emulator state
        if not paused:
            # Update keypad state
            state = state.replace(keypad=keypad_state)

            # Execute instructions (multiple per frame for realistic speed)
            instructions_per_frame = max(1, fps // 10)  # Roughly 600 Hz at 60 FPS

            for _ in range(instructions_per_frame):
                try:
                    # Use octax fetch and execute functions
                    state, instruction = fetch(state)
                    state = execute(state, int(instruction))
                    instruction_count += 1

                    # Check for program end or invalid PC (convert JAX array to int)
                    pc_value = int(state.pc)
                    if pc_value >= len(state.memory) - 1:
                        print("Program ended (PC out of bounds)")
                        paused = True
                        break

                except Exception as e:
                    print(f"Execution error at PC=0x{int(state.pc):03X}: {e}")
                    paused = True
                    break

            # Update timers at 60Hz - handle JAX arrays properly
            current_time = time.time()
            if current_time - last_timer_update >= 1.0 / 60:  # 60Hz timer update
                delay_val = int(state.delay_timer)
                sound_val = int(state.sound_timer)

                if delay_val > 0:
                    state = state.replace(delay_timer=jnp.asarray(delay_val - 1, dtype=jnp.uint8))
                if sound_val > 0:
                    state = state.replace(sound_timer=jnp.asarray(sound_val - 1, dtype=jnp.uint8))
                    # TODO: Add sound playing here when sound_timer > 0
                last_timer_update = current_time

        # Render display
        screen.fill(BLACK)

        # Draw CHIP-8 display using constants
        for y in range(SCREEN_HEIGHT):
            for x in range(SCREEN_WIDTH):
                if state.display[x, y]:
                    rect = pygame.Rect(
                        x * scale, y * scale,
                        scale, scale
                    )
                    pygame.draw.rect(screen, GREEN, rect)

        # Draw status info - convert JAX arrays to int for display
        font = pygame.font.Font(None, 24)
        status_text = f"PC: 0x{int(state.pc):03X} | I: 0x{int(state.I):03X} | Instructions: {instruction_count}"
        if paused:
            status_text += " | PAUSED"
        text_surface = font.render(status_text, True, WHITE)
        screen.blit(text_surface, (5, 5))

        # Show timer values - convert JAX arrays to int
        timer_text = f"Delay: {int(state.delay_timer)} | Sound: {int(state.sound_timer)}"
        timer_surface = font.render(timer_text, True, WHITE)
        screen.blit(timer_surface, (5, window_height - 25))

        # Show emulator mode
        mode_text = f"Mode: {'Modern' if state.modern_mode else 'Legacy'}"
        mode_surface = font.render(mode_text, True, WHITE)
        screen.blit(mode_surface, (5, 55))

        # Show pressed keys
        pressed_keys = [f"{i:X}" for i in range(16) if keypad_state[i]]
        if pressed_keys:
            key_text = f"Keys: {' '.join(pressed_keys)}"
            key_surface = font.render(key_text, True, WHITE)
            screen.blit(key_surface, (5, 30))

        pygame.display.flip()

    pygame.quit()
    return state


def run_emulator_with_rom(rom_filename: str, modern_mode=True, scale=10, fps=60, rng_seed=0):
    """
    Convenience function to run emulator with a ROM file

    Args:
        rom_filename: Path to ROM file
        modern_mode: Use modern CHIP-8 behavior (True) or legacy (False)
        scale: Display scale factor
        fps: Target FPS
        rng_seed: Random seed for RNG key
    """

    # Create initial state with specified mode using proper octax function
    rng_key = jax.random.PRNGKey(rng_seed)
    initial_state = create_state(rng_key).replace(modern_mode=modern_mode)

    print(f"Starting CHIP-8 Emulator ({'Modern' if modern_mode else 'Legacy'} mode)")

    final_state = run_interactive_emulator(
        initial_state,
        rom_filename,
        scale=scale,
        fps=fps
    )

    print("Emulator closed.")
    return final_state


def create_demo_rom(filename="demo.ch8"):
    """Create a simple demo ROM file for testing"""

    # Simple demo program: clear screen, wait for key, display key as sprite
    demo_program = [
        0x00, 0xE0,  # CLS - Clear screen
        0xA0, 0x50,  # LD I, 0x50 - Point to font data
        0xF0, 0x0A,  # LD V0, K - Wait for key press
        0xF0, 0x29,  # LD F, V0 - Set I to sprite for digit V0
        0x60, 0x20,  # LD V0, 32 - X coordinate
        0x61, 0x10,  # LD V1, 16 - Y coordinate
        0xD0, 0x15,  # DRW V0, V1, 5 - Draw sprite
        0x12, 0x04,  # JP 0x204 - Jump back to wait for key
    ]

    with open(filename, 'wb') as f:
        f.write(bytes(demo_program))

    print(f"Created demo ROM: {filename}")
    return filename


# Example usage functions
def demo_with_file():
    """Demo using a ROM file"""
    demo_file = create_demo_rom()
    run_emulator_with_rom(demo_file, modern_mode=True, scale=12)


def demo_legacy_mode():
    """Demo in legacy CHIP-8 mode"""
    demo_file = create_demo_rom()
    run_emulator_with_rom(demo_file, modern_mode=False, scale=10)


def demo_custom_state():
    """Demo with custom initial state"""
    # Create state with custom RNG seed
    rng_key = jax.random.PRNGKey(42)
    state = create_state(rng_key).replace(modern_mode=True)

    demo_file = create_demo_rom()
    run_interactive_emulator(state, demo_file, scale=10)


if __name__ == "__main__":
    # Run demo
    demo_with_file()