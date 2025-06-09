"""CHIP-8 rendering utilities for visualization."""
import time

import jax.numpy as jnp
import numpy as np
from typing import Tuple
import cv2

from PIL import Image

from octax import EmulatorState


def chip8_display_to_rgb(
    display: jnp.ndarray,
    scale: int = 8,
    on_color: Tuple[int, int, int] = (0, 255, 0),
    off_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Convert CHIP-8 boolean display to RGB array with optional upscaling.

    Args:
        display: Boolean array of shape (64, 32) representing CHIP-8 display
        scale: Upscaling factor for better visibility (default: 8x)
        on_color: RGB color for "on" pixels (default: green)
        off_color: RGB color for "off" pixels (default: black)

    Returns:
        RGB array of shape (height*scale, width*scale, 3) with uint8 values
    """
    pixels = np.array(display, dtype=np.bool_)

    # Original: (64 width, 32 height) -> Display: (32 height, 64 width)
    pixels = pixels.T
    height, width = pixels.shape

    rgb_frame = np.zeros((height, width, 3), dtype=np.uint8)

    rgb_frame[pixels] = on_color
    rgb_frame[~pixels] = off_color

    # Apply upscaling using nearest neighbor interpolation
    if scale > 1:
        rgb_frame = np.repeat(np.repeat(rgb_frame, scale, axis=0), scale, axis=1)

    return rgb_frame


def create_color_scheme(
    scheme: str = "octax",
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Get predefined color schemes for CHIP-8 rendering.

    Args:
        scheme: Color scheme name ("octax", "classic", "amber", "white", "blue")

    Returns:
        Tuple of (on_color, off_color) as RGB tuples
    """
    schemes = {
        "octax" : ((179, 102, 184), (45, 25, 61)),  # Octax theme
        "classic": ((0, 255, 0), (0, 0, 0)),  # Green on black
        "amber": ((255, 176, 0), (0, 0, 0)),  # Amber on black
        "white": ((255, 255, 255), (0, 0, 0)),  # White on black
        "blue": ((0, 255, 255), (0, 0, 64)),  # Cyan on dark blue
        "retro": ((255, 255, 0), (64, 0, 64)),  # Yellow on purple
    }

    if scheme not in schemes:
        raise ValueError(
            f"Unknown color scheme '{scheme}'. Available: {list(schemes.keys())}"
        )

    return schemes[scheme]


def batch_render(
    displays: jnp.ndarray, scale: int = 4, color_scheme: str = "octax"
) -> np.ndarray:
    """Render multiple CHIP-8 displays in a grid layout with transparent spacing.

    Args:
        displays: Array of shape (batch_size, 64, 32) with multiple displays
        scale: Upscaling factor (smaller for batch rendering)
        color_scheme: Color scheme name

    Returns:
        RGBA array showing all displays in a grid layout with transparent padding
    """
    batch_size = displays.shape[0]
    on_color, off_color = create_color_scheme(color_scheme)

    padding = 5  # space between displays in pixels

    # Calculate grid dimensions
    grid_cols = int(np.ceil(np.sqrt(batch_size)))
    grid_rows = int(np.ceil(batch_size / grid_cols))

    # Render individual displays and convert to RGBA
    rendered_displays = []
    for i in range(batch_size):
        rgb = chip8_display_to_rgb(displays[i], scale, on_color, off_color)
        rgba = np.concatenate([rgb, 255 * np.ones((*rgb.shape[:2], 1), dtype=np.uint8)], axis=-1)
        rendered_displays.append(rgba)

    # Pad with transparent displays if needed
    while len(rendered_displays) < grid_rows * grid_cols:
        transparent_display = np.zeros_like(rendered_displays[0])
        rendered_displays.append(transparent_display)

    # Arrange in grid with transparent padding
    display_height, display_width = rendered_displays[0].shape[:2]
    grid_height = grid_rows * display_height + (grid_rows - 1) * padding
    grid_width = grid_cols * display_width + (grid_cols - 1) * padding
    grid_image = np.zeros((grid_height, grid_width, 4), dtype=np.uint8)  # RGBA

    for i, rendered in enumerate(rendered_displays):
        row = i // grid_cols
        col = i % grid_cols
        y_start = row * (display_height + padding)
        x_start = col * (display_width + padding)
        y_end = y_start + display_height
        x_end = x_start + display_width
        grid_image[y_start:y_end, x_start:x_end] = rendered

    return grid_image


def create_video(
        state: EmulatorState,
        filename: str = None,
        fps: float = 60.0,
        scale: int = 8,
        color_scheme: str = "octax",
        persistence: bool = True,
        display: bool = False
) -> None:
    """Display and/or save CHIP-8 video with optional phosphor persistence.

    Args:
        state: EmulatorState with display shape (N, 64, 32)
        filename: If provided, save video to this MP4 file
        fps: Video frame rate
        scale: Upscaling factor
        color_scheme: Color scheme for rendering
        persistence: Enable phosphor screen simulation (smooth fading)
        display: If True, show video in window (press 'q' to quit, space to pause)
    """
    if state is None and display is False:
        return
    displays = np.array(state.display)
    if len(displays.shape) != 3 or displays.shape[1:] != (64, 32):
        raise ValueError(f"Expected display shape (N, 64, 32), got {displays.shape}")

    # Setup
    height, width = 32 * scale, 64 * scale
    on_color, off_color = np.array(create_color_scheme(color_scheme))

    # Video writer (if saving)
    writer = None
    if filename:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    # Display window (if showing)
    if display:
        window_name = "CHIP-8 Video (q=quit, space=pause)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Phosphor glow buffer
    glow = np.zeros((64, 32), dtype=np.float32) if persistence else None
    decay = 0.8

    # Frame timing for display
    frame_delay = 1.0 / fps if display else 0
    paused = False

    try:
        for i, frame_display in enumerate(displays):
            start_time = time.time()

            # Apply phosphor persistence or normal rendering
            if persistence:
                glow = glow * decay + frame_display.astype(np.float32)
                glow = np.clip(glow, 0.0, 1.0)
                pixel_values = glow.T  # (32, 64)
            else:
                pixel_values = frame_display.T.astype(np.float32)  # (32, 64)

            # Render frame with color interpolation
            frame = np.zeros((32, 64, 3), dtype=np.uint8)
            for c in range(3):
                frame[:, :, c] = off_color[c] + pixel_values * (on_color[c] - off_color[c])

            # Scale up and convert to BGR for OpenCV
            if scale > 1:
                frame = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Save frame
            if writer:
                writer.write(frame_bgr)

            # Display frame
            if display:
                # Add frame counter
                cv2.putText(frame_bgr, f"Frame {i + 1}/{len(displays)}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(window_name, frame_bgr)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord(' '):  # Space to pause/unpause
                    paused = not paused
                    if paused:
                        print("Paused - press space to continue, 'q' to quit")

                # Pause handling
                while paused:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord(' '):
                        paused = False
                        print("Resumed")
                        break
                    elif key == ord('q') or key == 27:
                        return

                # Frame rate timing
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        # Cleanup
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    # Report results
    if filename:
        duration = len(displays) / fps
        print(f"Video saved: {filename} ({len(displays)} frames, {fps} FPS, {duration:.1f}s)")
    if display:
        print("Display closed")