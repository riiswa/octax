import importlib
import os.path
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

from octax import EmulatorState
import jax.numpy as jnp

from octax.env import OctaxEnv


class EnvDef(ModuleType):
    rom_file: str
    score_fn: Callable[[EmulatorState], float | jnp.ndarray]
    terminated_fn: Callable[[EmulatorState], bool | jnp.ndarray]
    action_set: list
    metadata: dict
    startup_instructions: int


def get_rom_path(rom_filename):
    current = Path(__file__).parent.resolve()

    # Walk up until we find roms directory
    while current != current.parent:
        roms_path = current / 'roms' / rom_filename
        if roms_path.exists():
            return str(roms_path)
        current = current.parent

    # If not found, raise error with helpful message
    raise FileNotFoundError(f"ROM '{rom_filename}' not found. Make sure 'roms' directory exists at repository root.")


def create_environment(
    env_id: str,
    render_mode: Optional[str] = None,
    render_scale: int = 8,
    color_scheme: str = "classic",
    **kwargs,
):
    """Create an Octax CHIP-8 environment.

    Args:
        env_id: Environment identifier (e.g., "brix", "tetris")
        render_mode: Rendering mode ("rgb_array" or None)
        render_scale: Upscaling factor for rendered frames (default: 8x)
        color_scheme: Color scheme for rendering ("classic", "amber", "white", "blue", "retro")
        **kwargs: Additional parameters passed to OctaxEnv

    Returns:
        Tuple of (environment, metadata)
    """
    module: EnvDef = importlib.import_module(f"octax.environments.{env_id.replace('-', '_')}")

    return (
        OctaxEnv(
            rom_path=os.path.join(get_rom_path(module.rom_file)),
            score_fn=getattr(module, "score_fn", lambda _: 0),
            terminated_fn=getattr(module, "terminated_fn", lambda _: False),
            action_set=getattr(module, "action_set", None),
            startup_instructions=getattr(module, "startup_instructions", 0),
            custom_startup=getattr(module, "custom_startup", None),
            disable_delay=getattr(module, "disable_delay", False),
            render_mode=render_mode,
            render_scale=render_scale,
            color_scheme=color_scheme,
            **kwargs,
        ),
        module.metadata,
    )


def print_metadata(program: dict):
    """Print CHIP-8 program metadata with useful information only.

    Args:
        program: Program dictionary following CHIP-8 database schema
    """
    # Header
    title = program.get('title', 'Unknown Program')
    release = program.get('release', 'Unknown')
    print(f"🎮 {title} ({release})")

    # Authors
    if program.get('authors'):
        authors = ", ".join(program['authors'])
        print(f"   By: {authors}")

    # Description
    if program.get('description'):
        desc = program['description'].strip()
        if desc:
            print(f"   {desc}")

    # ROM info
    if program.get('roms'):
        for rom_hash, rom_info in program['roms'].items():
            filename = rom_info.get('file', 'Unknown ROM')
            print(f"   📁 {filename}")

            # Platform compatibility
            if rom_info.get('platforms'):
                platforms = " → ".join(rom_info['platforms'])
                print(f"      Platforms: {platforms}")

            # Controls (most important for games)
            if rom_info.get('keys'):
                keys = rom_info['keys']
                controls = []

                # Movement keys
                if any(k in keys for k in ['up', 'down', 'left', 'right']):
                    dirs = []
                    if 'up' in keys: dirs.append(f"↑{keys['up']}")
                    if 'down' in keys: dirs.append(f"↓{keys['down']}")
                    if 'left' in keys: dirs.append(f"←{keys['left']}")
                    if 'right' in keys: dirs.append(f"→{keys['right']}")
                    controls.append(" ".join(dirs))

                # Action keys
                if any(k in keys for k in ['a', 'b']):
                    actions = []
                    if 'a' in keys: actions.append(f"A={keys['a']}")
                    if 'b' in keys: actions.append(f"B={keys['b']}")
                    controls.append(" ".join(actions))

                if controls:
                    print(f"      Controls: {' | '.join(controls)}")

            # Important settings
            if rom_info.get('tickrate'):
                print(f"      Speed: {rom_info['tickrate']} cycles/frame")

            if rom_info.get('screenRotation', 0) != 0:
                print(f"      Rotation: {rom_info['screenRotation']}°")

    print()

if __name__ == "__main__":
    # Example
    import jax.numpy as jnp
    import jax
    import cv2
    import numpy as np
    import time

    env, metadata = create_environment("shooting-stars")

    print_metadata(metadata)

    def policy(rng: jax.random.PRNGKey, observation: jnp.ndarray):
        return jax.random.randint(rng, (), 0, env.num_actions)

    @jax.jit
    def rollout(rng):
        def env_step(carry, _):
            rng, state, observation = carry
            rng, rng_action, rng_reset = jax.random.split(rng, 3)
            action = policy(rng_action, observation)
            next_state, next_observation, reward, terminated, truncated, info = env.step(state, action)
            jax.debug.print("{}", reward)

            next_state, next_observation, info = jax.lax.cond(terminated, env.reset, lambda _: (next_state, next_observation, info), rng_reset)

            return (rng, next_state, next_observation), observation

        rng, rng_reset = jax.random.split(rng)
        state, observation, info = env.reset(rng_reset)
        return jax.lax.scan(env_step, (rng, state, observation), length=env.max_num_steps_per_episodes)

    rng = jax.random.PRNGKey(0)

    # Measure compilation time
    start_compile = time.time()
    compiled = jax.block_until_ready(rollout.lower(rng).compile())
    end_compile = time.time()

    print("Compilation time (s):", end_compile - start_compile)

    # Measure execution time
    start_exec = time.time()
    final_state, observations = jax.block_until_ready(compiled(rng))
    end_exec = time.time()

    print("Execution time (s):", end_exec - start_exec)

    frames = np.array(observations[:, -1]) * 255  # Convert to 0-255 range
    frames = frames.astype(np.uint8)

    # Display frames
    for i, frame in enumerate(frames):
        # Resize for visibility
        display_frame = cv2.resize(frame.T, (640, 320),
                                   interpolation=cv2.INTER_NEAREST)

        cv2.putText(display_frame, f"{i + 1}/{len(frames)}", (540, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('CHIP-8 Display', display_frame)

        # Exit on 'q' or ESC, pause on spacebar
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord(' '):  # Spacebar to pause
            cv2.waitKey(0)

    cv2.destroyAllWindows()
