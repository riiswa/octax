import importlib
import os.path
import re
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

from octax import EmulatorState
import jax.numpy as jnp

from octax.env import OctaxEnv
from octax.rendering import create_video


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
    env_id = env_id.replace('-', '_')
    match = re.match(r'^(.*?)(\d+)$', env_id)
    if match:
        rom_file = env_id + ".ch8"
        env_id = match.group(1)
        have_level = True
    else:
        have_level = False
    module: EnvDef = importlib.import_module(f"octax.environments.{env_id}")
    if have_level:
        module.__setattr__("rom_file", rom_file)

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

