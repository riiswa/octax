"""
Clean CHIP-8 emulator with simple score detection
"""

import pygame
import jax
import jax.numpy as jnp
import time
import numpy as np
from collections import deque
from octax.state import create_state
from octax import execute, fetch, load_rom

# Modern key mapping
KEY_MAP = {
    pygame.K_1: 0x1, pygame.K_2: 0x2, pygame.K_3: 0x3, pygame.K_4: 0x4,
    pygame.K_5: 0x5, pygame.K_6: 0x6, pygame.K_7: 0x7, pygame.K_8: 0x8,
    pygame.K_9: 0x9, pygame.K_0: 0x0,
    pygame.K_UP: 0x2, pygame.K_DOWN: 0x8, pygame.K_LEFT: 0x4, pygame.K_RIGHT: 0x6,
    pygame.K_w: 0x2, pygame.K_s: 0x8, pygame.K_a: 0x4, pygame.K_d: 0x6,
    pygame.K_SPACE: 0x5,
    pygame.K_q: 0xA, pygame.K_e: 0xB, pygame.K_t: 0xC,
    pygame.K_y: 0xD, pygame.K_u: 0xE, pygame.K_i: 0xF
}


class SimpleDetector:
    """Lightweight register tracking for authentic performance"""

    def __init__(self):
        self.register_history = [[] for _ in range(16)]  # Store changes for each register
        self.last_values = [0] * 16  # Track last known value for each register
        self.bcd_registers = set()
        self.last_print_time = 0

    def detect_bcd(self, state, instruction, instruction_count):
        """Detect BCD operations - strongest score indicator"""
        if (instruction & 0xF0FF) == 0xF033:
            register = (instruction & 0x0F00) >> 8
            value = int(state.V[register])
            address = int(state.I)
            self.bcd_registers.add(register)
            print(f"🎯 BCD! V{register:X} = {value} -> MEM[0x{address:03X}]")

    def track_changes(self, state):
        """Lightweight change tracking"""
        for reg in range(16):
            current_value = int(state.V[reg])
            last_value = self.last_values[reg]

            if current_value != last_value:
                # Store only the value (no instruction count for performance)
                self.register_history[reg].append(current_value)
                # Keep only last 5 changes
                if len(self.register_history[reg]) > 5:
                    self.register_history[reg].pop(0)
                self.last_values[reg] = current_value

    def print_all_registers(self, state, instruction_count):
        """Print all 16 registers with last 5 changes"""
        current_time = time.time()
        if current_time - self.last_print_time < 30.0:
            return

        self.last_print_time = current_time
        print(f"\n📊 ALL REGISTERS (Instruction #{instruction_count}):")

        # Print all 16 registers
        for reg in range(16):
            current = int(state.V[reg])
            changes = self.register_history[reg]

            # Build change history string
            if len(changes) >= 2:
                hist_display = f" [{' -> '.join(str(v) for v in changes)}]"

                # Simple trend detection
                if len(changes) >= 3:
                    increasing = sum(1 for i in range(1, len(changes)) if changes[i] > changes[i-1])
                    decreasing = sum(1 for i in range(1, len(changes)) if changes[i] < changes[i-1])

                    if increasing >= len(changes) // 2:
                        trend = "📈"
                    elif decreasing >= len(changes) // 2:
                        trend = "📉"
                    else:
                        trend = "📊"
                else:
                    trend = "📊"
            elif len(changes) == 1:
                hist_display = f" [changed to {changes[0]}]"
                trend = "📊"
            else:
                hist_display = " [no changes]"
                trend = "📊"

            # Mark BCD registers
            bcd_mark = " 🎯" if reg in self.bcd_registers else ""

            print(f"  V{reg:X}: {current:3d}{hist_display} {trend}{bcd_mark}")

        if self.bcd_registers:
            bcd_regs = ", ".join(f"V{r:X}" for r in sorted(self.bcd_registers))
            print(f"🔢 BCD Registers: {bcd_regs}")

        print("-" * 60)


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


def run_emulator(rom_filename,modern_mode=True, scale=8, ipf=17):
    """Main emulator loop - authentic CHIP-8 settings"""

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((64 * scale, 32 * scale))
    pygame.display.set_caption("CHIP-8 Score Detective")
    clock = pygame.time.Clock()

    # Create CHIP-8 state
    rng_key = jax.random.PRNGKey(0)
    state = create_state(rng_key).replace(modern_mode=modern_mode)

    try:
        state = load_rom(state, rom_filename)
        print(f"✅ Loaded: {rom_filename}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Game state
    keypad_state = jnp.zeros(16, dtype=jnp.bool_)
    instruction_count = 0
    running = True
    paused = False
    show_debug = True

    # Detection
    detector = SimpleDetector()
    snapshot_timer = 0
    start_time = time.time()

    # Frame rate tracking
    frame_count = 0
    fps_start_time = time.time()
    current_fps = 60

    print("🎮 Controls: ESC=Quit, P=Pause, R=Reset, ±=Speed, D=Debug")
    print("🎯 Watching for BCD operations and register patterns...")

    while running:
        clock.tick(60)

        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        if current_time - fps_start_time >= 1.0:
            current_fps = frame_count / (current_time - fps_start_time)
            frame_count = 0
            fps_start_time = current_time

        # Handle input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_d:
                    show_debug = not show_debug
                elif event.key == pygame.K_r:
                    # Reset
                    state = create_state(rng_key).replace(modern_mode=modern_mode)
                    state = load_rom(state, rom_filename)
                    keypad_state = jnp.zeros(16, dtype=jnp.bool_)
                    instruction_count = 0
                    detector = SimpleDetector()
                    print("🔄 Reset")
                elif event.key == pygame.K_EQUALS:
                    ipf = min(100, ipf + 3)
                    print(f"⚡ Speed: {ipf} IPF")
                elif event.key == pygame.K_MINUS:
                    ipf = max(3, ipf - 3)
                    print(f"🐌 Speed: {ipf} IPF")
                elif event.key in KEY_MAP:
                    keypad_state = keypad_state.at[KEY_MAP[event.key]].set(True)
            elif event.type == pygame.KEYUP:
                if event.key in KEY_MAP:
                    keypad_state = keypad_state.at[KEY_MAP[event.key]].set(False)

        # Update emulator
        if not paused:
            state = state.replace(keypad=keypad_state)

            # Update timers at 60 Hz (once per frame)
            if int(state.delay_timer) > 0:
                state = state.replace(delay_timer=jnp.asarray(int(state.delay_timer) - 1, dtype=jnp.uint8))
            if int(state.sound_timer) > 0:
                state = state.replace(sound_timer=jnp.asarray(int(state.sound_timer) - 1, dtype=jnp.uint8))

            # Execute instructions at authentic CHIP-8 speed (~600 Hz)
            for i in range(ipf):
                try:
                    state, instruction = fetch(state)
                    # Only check BCD every few instructions for performance
                    if i % 5 == 0:
                        detector.detect_bcd(state, int(instruction), instruction_count)
                    state = execute(state, int(instruction))
                    instruction_count += 1

                except Exception as e:
                    print(f"💥 Error: {e}")
                    paused = True
                    break

            # Track changes once per frame (not per instruction)
            detector.track_changes(state)

            # Print status every 1800 instructions (~3 seconds at 600 Hz)
            snapshot_timer += ipf
            if snapshot_timer >= 1800:
                detector.print_all_registers(state, instruction_count)
                snapshot_timer = 0

        # Render
        screen.fill((0, 0, 0))
        for y in range(32):
            for x in range(64):
                if state.display[x, y]:
                    rect = pygame.Rect(x * scale, y * scale, scale, scale)
                    pygame.draw.rect(screen, (0, 255, 0), rect)

        # Draw debug overlay if enabled
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
                f"CPU: {ips:.0f} Hz (target: ~600 Hz)",
                f"IPF: {ipf}",
                f"FPS: {current_fps:.1f} (target: 60)",
                f"Status: {'PAUSED' if paused else 'RUNNING'}"
            ]

            draw_overlay_text(screen, debug_lines, (5, 5), font_small, alpha=100)

            # Timer info (top-right)
            timer_lines = [
                f"Delay: {int(state.delay_timer)}",
                f"Sound: {int(state.sound_timer)}"
            ]

            draw_overlay_text(screen, timer_lines, (64 * scale - 80, 5), font_small, alpha=100)

            # Register info (bottom-left) - highlight BCD registers
            reg_lines = []
            for i in range(0, 16, 4):
                reg_parts = []
                for j in range(i, min(i + 4, 16)):
                    reg_value = f"{int(state.V[j]):02X}"
                    if j in detector.bcd_registers:
                        reg_parts.append(f"*V{j:X}:{reg_value}*")  # Mark BCD registers
                    else:
                        reg_parts.append(f"V{j:X}:{reg_value}")
                reg_lines.append(" ".join(reg_parts))

            draw_overlay_text(screen, reg_lines, (5, 32 * scale - 80), font_tiny, alpha=80)

            # BCD operations count (middle-right)
            if detector.bcd_registers:
                bcd_regs = ", ".join(f"V{r:X}" for r in sorted(detector.bcd_registers))
                bcd_lines = [
                    "BCD Registers:",
                    bcd_regs
                ]
                draw_overlay_text(screen, bcd_lines, (64 * scale - 120, 80), font_tiny,
                                text_color=(255, 255, 0), alpha=120)

            # Active keys (bottom-right)
            pressed_keys = [f"{i:X}" for i in range(16) if keypad_state[i]]
            if pressed_keys:
                key_lines = [
                    "Keys: " + " ".join(pressed_keys)
                ]
                draw_overlay_text(screen, key_lines, (64 * scale - 100, 32 * scale - 25), font_tiny, alpha=150)

        # Simple status when debug is off or paused
        elif paused or not show_debug:
            font = pygame.font.Font(None, 24)
            if paused:
                text = font.render("PAUSED - P to resume", True, (255, 255, 0))
            else:
                text = font.render("D for Debug", True, (255, 255, 0))
            screen.blit(text, (10, 10))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    # Authentic CHIP-8 settings: 10 IPF × 60 FPS = 600 Hz CPU speed
    run_emulator("Chip8-Database/Chip8-Games/Deflap (fix)(hitcherland)(2015).ch8", True, scale=8, ipf=10)