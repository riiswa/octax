# Changelog

All notable changes to Octax are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and the project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.1.1] — 2025

### Added
- **22 game environments** across five genre categories
- **LLM-generated curriculum environments** (`target_shooter1/2/3`)
- Gymnax compatibility wrapper (`OctaxGymnaxWrapper`)
- PPO and PQN agent implementations (`octax/agents/`)
- `create_video` rendering utility with phosphor persistence effect
- Batch rendering (`batch_render`) for visualising parallel environments
- Six built-in colour schemes: `octax`, `classic`, `amber`, `white`, `blue`, `retro`
- `play.py` interactive emulator with real-time register inspection and BCD detection
- `scan_with_progress` / `fori_loop_with_progress` utilities for JAX training loops
- CI pipelines for Linux, macOS (Apple Silicon M1 + M2), and Windows
- Sphinx documentation with MyST parser

### Changed
- Modified Worm V4 ROM: added game-over flag patch for clean RL termination
- Modified Cavern ROMs: replaced survival scoring with leftward-progress reward
- Modified Space Flight ROMs: single-life mode for faster episode cycling
- Modified Flight Runner ROM: added game-over flag and score counter

---

## [0.1.0] — 2025

Initial public release.

### Added
- Core CHIP-8 emulator in JAX (`octax/emulator.py`, `octax/state.py`, `octax/decode.py`)
- Full instruction set coverage (35 instructions)
- Modern and legacy mode support for shift / store / jump-with-offset quirks
- `OctaxEnv` — JAX-native RL environment wrapper with JIT-compatible reset/step
- `create_environment` factory for all built-in games
- ROM loading from `roms/` directory
- Frame-skip observation stacking
- `chip8_display_to_rgb` rendering
- `print_metadata` utility
- Comprehensive test suite (`tests/`)
