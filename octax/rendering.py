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

