"""CHIP-8 rendering utilities for visualization."""

import jax.numpy as jnp
import numpy as np
from typing import Tuple


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
    scheme: str = "classic",
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Get predefined color schemes for CHIP-8 rendering.

    Args:
        scheme: Color scheme name ("classic", "amber", "white", "blue")

    Returns:
        Tuple of (on_color, off_color) as RGB tuples
    """
    schemes = {
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


def render_with_info(
    display: jnp.ndarray,
    score: float = 0,
    lives: int = 0,
    scale: int = 8,
    color_scheme: str = "classic",
) -> np.ndarray:
    """Render CHIP-8 display with optional game information overlay.

    Args:
        display: Boolean array representing CHIP-8 display
        score: Current game score to display
        lives: Current lives/health to display
        scale: Upscaling factor
        color_scheme: Color scheme name

    Returns:
        RGB array with game display and info overlay
    """
    # Get base display
    on_color, off_color = create_color_scheme(color_scheme)
    rgb_frame = chip8_display_to_rgb(display, scale, on_color, off_color)

    # Add score and lives overlay

    return rgb_frame


def batch_render(
    displays: jnp.ndarray, scale: int = 4, color_scheme: str = "classic"
) -> np.ndarray:
    """Render multiple CHIP-8 displays in a grid layout.

    Args:
        displays: Array of shape (batch_size, 64, 32) with multiple displays
        scale: Upscaling factor (smaller for batch rendering)
        color_scheme: Color scheme name

    Returns:
        RGB array showing all displays in a grid layout
    """
    batch_size = displays.shape[0]
    on_color, off_color = create_color_scheme(color_scheme)

    # Calculate grid dimensions
    grid_cols = int(np.ceil(np.sqrt(batch_size)))
    grid_rows = int(np.ceil(batch_size / grid_cols))

    # Render individual displays
    rendered_displays = []
    for i in range(batch_size):
        rendered = chip8_display_to_rgb(displays[i], scale, on_color, off_color)
        rendered_displays.append(rendered)

    # Pad with black displays if needed
    while len(rendered_displays) < grid_rows * grid_cols:
        black_display = np.zeros_like(rendered_displays[0])
        rendered_displays.append(black_display)

    # Arrange in grid
    display_height, display_width = rendered_displays[0].shape[:2]
    grid_image = np.zeros(
        (grid_rows * display_height, grid_cols * display_width, 3), dtype=np.uint8
    )

    for i, rendered in enumerate(rendered_displays):
        row = i // grid_cols
        col = i % grid_cols
        y_start = row * display_height
        y_end = y_start + display_height
        x_start = col * display_width
        x_end = x_start + display_width
        grid_image[y_start:y_end, x_start:x_end] = rendered

    return grid_image
