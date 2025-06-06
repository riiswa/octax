"""
Minimal CHIP-8 pygame loop using octax functions only
"""

import pygame
import jax
import jax.numpy as jnp
import time
from octax.state import create_state
from octax import execute, fetch, load_rom

# CHIP-8 keypad mapping
KEY_MAP = {
    pygame.K_1: 0x1, pygame.K_2: 0x2, pygame.K_3: 0x3, pygame.K_4: 0xC,
    pygame.K_q: 0x4, pygame.K_w: 0x5, pygame.K_e: 0x6, pygame.K_r: 0xD,
    pygame.K_a: 0x7, pygame.K_s: 0x8, pygame.K_d: 0x9, pygame.K_f: 0xE,
    pygame.K_z: 0xA, pygame.K_x: 0x0, pygame.K_c: 0xB, pygame.K_v: 0xF
}


def draw_overlay_text(surface, text_lines, position, font, bg_color=(0, 0, 0), text_color=(255, 255, 255), alpha=120):
    """Draw text with semi-transparent background overlay"""
    if not text_lines:
        return

    # Calculate overlay size
    line_height = font.get_height()
    max_width = max(font.size(line)[0] for line in text_lines)
    overlay_height = len(text_lines) * line_height + 8
    overlay_width = max_width + 16

    # Create semi-transparent overlay
    overlay = pygame.Surface((overlay_width, overlay_height))
    overlay.set_alpha(alpha)
    overlay.fill(bg_color)
    surface.blit(overlay, position)

    # Draw text lines
    x, y = position
    for i, line in enumerate(text_lines):
        text_surface = font.render(line, True, text_color)
        surface.blit(text_surface, (x + 8, y + 4 + i * line_height))


def run_emulator(rom_filename, modern_mode=True, scale=10, fps=60, speed_multiplier=1.5):
    """Run CHIP-8 emulator with pygame"""

    # Initialize pygame
    pygame.init()
    window_width = 64 * scale
    window_height = 32 * scale
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("CHIP-8 Emulator - Octax")
    clock = pygame.time.Clock()

    # Create initial state using octax
    rng_key = jax.random.PRNGKey(0)
    state = create_state(rng_key).replace(modern_mode=modern_mode)

    # Load ROM using octax function
    try:
        state = load_rom(state, rom_filename)
        print(f"Loaded ROM: {rom_filename}")
    except Exception as e:
        print(f"Error loading ROM: {e}")
        pygame.quit()
        return None

    # Initialize tracking variables
    keypad_state = jnp.zeros(16, dtype=jnp.bool_)
    instruction_count = 0
    cycles_per_frame = max(1, fps // 10)  # ~600 Hz
    start_time = time.time()
    last_timer_update = time.time()

    running = True
    paused = False
    show_debug = True  # Toggle with 'D' key

    print("Controls: ESC=Quit, SPACE=Pause, D=Debug, Keys=1234/QWER/ASDF/ZXCV")

    while running:
        frame_start = time.time()
        clock.tick(fps)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_d:
                    show_debug = not show_debug
                elif event.key == pygame.K_r:
                    # Reset emulator
                    state = create_state(rng_key).replace(modern_mode=modern_mode)
                    state = load_rom(state, rom_filename)
                    keypad_state = jnp.zeros(16, dtype=jnp.bool_)
                    instruction_count = 0
                    start_time = time.time()
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    # Increase speed
                    speed_multiplier = min(5.0, speed_multiplier + 0.25)
                    print(f"Speed: {speed_multiplier:.2f}x")
                elif event.key == pygame.K_MINUS:
                    # Decrease speed
                    speed_multiplier = max(0.25, speed_multiplier - 0.25)
                    print(f"Speed: {speed_multiplier:.2f}x")
                elif event.key in KEY_MAP:
                    key_index = KEY_MAP[event.key]
                    keypad_state = keypad_state.at[key_index].set(True)
            elif event.type == pygame.KEYUP:
                if event.key in KEY_MAP:
                    key_index = KEY_MAP[event.key]
                    keypad_state = keypad_state.at[key_index].set(False)

        # Update emulator
        if not paused:
            # Update keypad - do this BEFORE instruction execution for responsiveness
            state = state.replace(keypad=keypad_state)

            # Calculate actual cycles based on speed multiplier
            actual_cycles = int(cycles_per_frame * speed_multiplier)

            # Execute instructions at high speed
            for _ in range(actual_cycles):
                try:
                    state, instruction = fetch(state)
                    state = execute(state, int(instruction))
                    instruction_count += 1

                    # Check for halt or infinite loop detection
                    if instruction_count > 1000000:  # Safety check
                        if instruction_count % 100000 == 0:
                            print(f"High instruction count: {instruction_count}")

                except Exception as e:
                    print(f"Execution error at PC=0x{int(state.pc):03X}, instruction=0x{instruction:04X}: {e}")
                    paused = True
                    break

            # Update timers at 60Hz
            current_time = time.time()
            if current_time - last_timer_update >= 1.0 / 60:
                delay_val = int(state.delay_timer)
                sound_val = int(state.sound_timer)

                if delay_val > 0:
                    state = state.replace(delay_timer=jnp.asarray(delay_val - 1, dtype=jnp.uint8))
                if sound_val > 0:
                    state = state.replace(sound_timer=jnp.asarray(sound_val - 1, dtype=jnp.uint8))

                last_timer_update = current_time

        # Render
        screen.fill((0, 0, 0))

        # Draw CHIP-8 display
        for y in range(32):
            for x in range(64):
                if state.display[x, y]:
                    rect = pygame.Rect(x * scale, y * scale, scale, scale)
                    pygame.draw.rect(screen, (0, 255, 0), rect)

        # Draw debug info if enabled
        if show_debug:
            font_small = pygame.font.Font(None, 18)
            font_tiny = pygame.font.Font(None, 16)

            # Main debug info (top-left)
            runtime = time.time() - start_time
            ips = instruction_count / runtime if runtime > 0 else 0

            debug_lines = [
                f"PC: 0x{int(state.pc):03X}",
                f"I: 0x{int(state.I):03X}",
                f"Instructions: {instruction_count}",
                f"Speed: {ips:.0f} Hz",
                f"Mode: {'Modern' if state.modern_mode else 'Legacy'}",
                f"Status: {'PAUSED' if paused else 'RUNNING'}"
            ]

            draw_overlay_text(screen, debug_lines, (5, 5), font_small, alpha=100)

            # Timer info (top-right)
            timer_lines = [
                f"Delay: {int(state.delay_timer)}",
                f"Sound: {int(state.sound_timer)}"
            ]

            draw_overlay_text(screen, timer_lines, (window_width - 80, 5), font_small, alpha=100)

            # Register info (bottom-left)
            reg_lines = []
            for i in range(0, 16, 4):
                reg_line = " ".join(f"V{j:X}:{int(state.V[j]):02X}" for j in range(i, min(i + 4, 16)))
                reg_lines.append(reg_line)

            draw_overlay_text(screen, reg_lines, (5, window_height - 80), font_tiny, alpha=80)

            # Active keys (bottom-right, only if keys pressed)
            pressed_keys = [f"{i:X}" for i in range(16) if keypad_state[i]]
            if pressed_keys:
                key_lines = [
                    "Keys: " + " ".join(pressed_keys)
                ]
                draw_overlay_text(screen, key_lines, (window_width - 100, window_height - 25), font_tiny, alpha=150)

        # Minimal status when debug is off
        else:
            font_tiny = pygame.font.Font(None, 16)
            minimal_status = f"{'⏸' if paused else '▶'} {instruction_count // 1000}k"

            draw_overlay_text(screen, [minimal_status], (5, 5), font_tiny, alpha=80)

        pygame.display.flip()

    pygame.quit()
    return state


if __name__ == "__main__":
    run_emulator("test_opcode.ch8", modern_mode=True)