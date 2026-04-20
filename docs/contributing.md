# Contributing to Octax

Thank you for your interest in contributing! Octax welcomes bug fixes, new game environments, documentation improvements, and performance enhancements.

## Getting Started

### Fork and clone

```bash
git clone https://github.com/riiswa/octax.git
cd octax
```

### Set up a development environment

```bash
pip install -e ".[all]"
```

This installs the package in editable mode along with all optional dependencies (training, GUI, dev, docs).

### Verify your setup

```bash
pytest tests/
```

All tests should pass before you start working.

---

## Types of Contributions

### Bug Reports

Open an issue on GitHub. Please include:

- A minimal reproducible example
- The full traceback
- Your platform (`uname -a` on Linux/macOS, `ver` on Windows)
- Python and JAX versions (`python --version`, `python -c "import jax; print(jax.__version__)"`)

### Bug Fixes

1. Open an issue (or comment on an existing one) so we know you are working on it
2. Create a branch: `git checkout -b fix/brief-description`
3. Write a test that fails before your fix and passes after
4. Submit a pull request referencing the issue

### New Game Environments

See the [Custom Game Environments tutorial](tutorials/custom_environments.md) for the full workflow. Before opening a PR, confirm:

- [ ] ROM is public domain or appropriately licensed (link the source)
- [ ] `score_fn` and `terminated_fn` are tested over at least one full episode
- [ ] Environment module is documented in `docs/environments/games.md`
- [ ] A test exists in `tests/`
- [ ] The game renders correctly with `create_video` (visual sanity check)

### Performance Improvements

JAX-specific optimisations (e.g., reducing traced operations, improving memory layout, eliminating Python-side loops) are especially welcome. Please include a benchmark comparing before/after using the existing performance script or a simple `jax.block_until_ready` timing block.

### Documentation

Documentation lives in `docs/` and is built with Sphinx + MyST. To build locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build
open docs/_build/index.html
```

---

## Pull Request Checklist

Before opening a PR, make sure the following are true:

- [ ] `pytest tests/` passes with no new failures
- [ ] New functionality has accompanying tests
- [ ] Docstrings follow the existing Google-style format
- [ ] Type hints are present on all public functions
- [ ] `docs/` is updated if the PR adds or changes user-facing behaviour
- [ ] The PR description explains *what* changed and *why*

---

## Code Style

Octax does not enforce a strict formatter, but please follow the patterns already present in the codebase:

- 4-space indentation
- Descriptive variable names (`decoded_instruction`, not `di`)
- Keep functions short — if a function exceeds ~60 lines, consider splitting it
- Prefer JAX-native operations over Python loops wherever possible

---

## Instruction Implementation Guide

If you are adding or fixing a CHIP-8 instruction, the implementation lives in `octax/instructions/`. Each file handles a logical group:

| File | Instructions |
|---|---|
| `system.py` | `00E0` (clear), `00EE` (return) |
| `control_flow.py` | `1NNN`, `2NNN`, `3–5XNN`, `9XY0`, `BNNN`, `EX9E/A1` |
| `memory.py` | `6XNN`, `7XNN`, `ANNN`, `CXNN` |
| `alu.py` | `8XYN` (all ALU variants) |
| `display.py` | `DXYN` |
| `misc.py` | `FX07/0A/15/18/1E/29/33/55/65` |

All instruction handlers have the same signature:

```python
def execute_my_instruction(
    state: EmulatorState,
    instruction: DecodedInstruction,
) -> EmulatorState:
    ...
```

They must be **pure functions** — no side effects, no mutation. Return a new state via `state.replace(...)`.

---

## Questions?

Open a [GitHub Discussion](https://github.com/riiswa/octax/discussions) or file an issue with the `question` label.
