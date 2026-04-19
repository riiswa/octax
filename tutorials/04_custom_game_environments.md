# Creating Custom Game Environments

Octax supports adding custom CHIP-8 games as reinforcement learning environments. This tutorial covers analyzing games, understanding their mechanics, and implementing environment definitions.

## Understanding Game Environment Structure

Every Octax game environment follows a consistent pattern. Let's examine the structure by looking at a simple existing game:

```python
from octax.environments import pong

# Every environment module has these components:
print("ROM file:", pong.rom_file)
print("Action set:", pong.action_set)
print("Metadata keys:", pong.metadata.keys())

# And these functions:
print("Has score_fn:", hasattr(pong, 'score_fn'))
print("Has terminated_fn:", hasattr(pong, 'terminated_fn'))
```

Each game environment is defined by:
- **ROM file**: The actual CHIP-8 game binary
- **Score function**: How to extract the current score from game state
- **Termination function**: When the game/episode ends
- **Action set**: Which CHIP-8 keys the game actually uses
- **Metadata**: Human-readable information about the game

The score and termination functions analyze the CHIP-8 emulator state to extract game information.

## The Game Analysis Process

Before creating an environment, you need to understand how your game works. Octax provides tools to help with this analysis process:

- [ ] TODO

## Creating Your First Custom Environment

- [ ] TODO