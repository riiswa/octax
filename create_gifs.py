"""
Generate one GIF per environment using a random policy.
Usage: python create_gifs.py
Output: docs/imgs/<env_name>.gif
"""

import os
import jax
import jax.numpy as jnp
from PIL import Image
from octax.environments import create_environment
from octax.rendering import chip8_display_to_rgb, create_color_scheme

os.makedirs("docs/_static/imgs", exist_ok=True)

ENVS = [
    # Single environments
    "airplane",
    "blinky",
    "brix",
    "deep",
    "filter",
    "flight_runner",
    "missile",
    "pong",
    "rocket",
    "shooting_stars",
    "spacejam",
    "squash",
    "submarine",
    "tank",
    "tetris",
    "ufo",
    "vertical_brix",
    "wipe_off",
    "worm",
    # Cavern levels (1-7)
    "cavern1",
    "cavern2",
    "cavern3",
    "cavern4",
    "cavern5",
    "cavern6",
    "cavern7",
    # Space Flight levels (1-10)
    "space_flight1",
    "space_flight2",
    "space_flight3",
    "space_flight4",
    "space_flight5",
    "space_flight6",
    "space_flight7",
    "space_flight8",
    "space_flight9",
    "space_flight10",
    # Target Shooter levels (1-3)
    "target_shooter1",
    "target_shooter2",
    "target_shooter3",
]

N_FRAMES = 150   # number of steps to record
FRAME_SKIP = 2   # only save every Nth frame to keep GIF small
FPS = 20         # GIF playback speed
SCALE = 4        # pixel upscale factor
COLOR_SCHEME = "octax"


def collect_frames(env, rng, n_frames):
    on_color, off_color = create_color_scheme(COLOR_SCHEME)
    frames = []

    state, obs, info = env.reset(rng)

    for i in range(n_frames):
        rng, rng_action = jax.random.split(rng)
        action = jax.random.randint(rng_action, (), 0, env.num_actions)
        state, obs, reward, terminated, truncated, info = env.step(state, action)

        if terminated or truncated:
            rng, rng_reset = jax.random.split(rng)
            state, obs, info = env.reset(rng_reset)

        if i % FRAME_SKIP == 0:
            frame = chip8_display_to_rgb(
                state.display,
                scale=SCALE,
                on_color=on_color,
                off_color=off_color,
            )
            frames.append(Image.fromarray(frame))

    return frames


def save_gif(frames, path, fps=FPS):
    duration_ms = int(1000 / fps)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=True,
    )


for env_name in ENVS:
    print(f"Generating {env_name}...", end=" ", flush=True)
    try:
        env, metadata = create_environment(env_name)
        rng = jax.random.PRNGKey(42)
        frames = collect_frames(env, rng, N_FRAMES)
        out_path = f"docs/_static/imgs/{env_name}.gif"
        save_gif(frames, out_path)
        print(f"saved → {out_path}")
    except Exception as e:
        print(f"FAILED ({e})")

print("\nDone.")
