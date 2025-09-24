"""
TARGET SHOOTER: LLM-Generated RL Training Environment
======================================================
Fully AI-generated CHIP-8 game environment for reinforcement learning research
Created: 2024 - Designed and implemented entirely by Large Language Model
Purpose: Progressive difficulty levels for curriculum learning in RL agents

ENVIRONMENT OVERVIEW:
This is a completely LLM-generated game environment specifically crafted for
RL research. The game features three progressive difficulty levels to enable
curriculum learning, where agents can master basic skills before advancing to
more complex challenges.

KEY RL FEATURES:
- Deterministic game mechanics for reproducible training
- Clear reward signal via score register V[2]
- Binary termination flag in register V[3]
- Observable game state through CHIP-8 registers
- Action space: WASD movement + E to shoot (keys 5,7,8,9,6)

LEVEL DESCRIPTIONS:
==================

LEVEL 1 - STATIC TARGETS:
- Targets appear randomly but don't move
- No time pressure
- Pure aiming skill development
- Ideal for initial policy learning

LEVEL 2 - TIME-LIMITED TARGETS:
- Static targets with ~3 second timeout
- Introduces time pressure and decision making
- Missed targets count toward game end
- Tests reaction time and prioritization

LEVEL 3 - MOVING TARGETS WITH TIME LIMIT:
- Targets move and bounce off walls
- Time limit still applies
- Maximum difficulty requiring predictive aiming
- Tests advanced motor planning and interception

REGISTER MAP (Consistent Across All Levels):
- V[0]: Crosshair X position
- V[1]: Crosshair Y position
- V[2]: Score (RL reward signal)
- V[3]: Game over flag (0=playing, 1=terminated)
- V[4]: Target X position
- V[5]: Target Y position
- V[6]: Target active flag
- V[A]: Targets total (Level 2-3) or hits (Level 1)
- V[C]: Target timer (Level 2-3 only)
- V[D]: Target X velocity (Level 3 only)
- V[E]: Target Y velocity (Level 3 only)
"""

from octax import EmulatorState

# Define rom files for each level
# Level 1: rom_file = "target_shooter_level1.ch8"
# Level 2: rom_file = "target_shooter_level2.ch8"
# Level 3: rom_file = "target_shooter_level3.ch8"

def score_fn(state: EmulatorState) -> float:
    """
    Extract score from register V[2]
    Score increments by 1 for each successful hit
    Range: 0-10 points
    """
    return state.V[2]

def terminated_fn(state: EmulatorState) -> bool:
    """
    Check game termination flag in register V[3]
    Game ends after 10 total targets (hit or missed in levels 2-3)
    """
    return state.V[3] == 1

# CHIP-8 key mapping for controls
# W=5 (up), A=7 (left), S=8 (down), D=9 (right), E=6 (shoot)
action_set = [5, 7, 8, 9, 6]

# Allow initial game setup
startup_instructions = 100

metadata = {
    "title": "Target Shooter - LLM-Generated RL Environment",
    "release": "2024",
    "authors": ["Fully LLM-Generated Environment"],
    "description": """
Target Shooter - AI-Generated Training Environment
===================================================

This CHIP-8 game is a complete LLM-generated environment designed 
specifically for reinforcement learning research. Every aspect - from
game mechanics to code implementation - was created by an AI language
model to serve as a benchmark for RL agent training.

WHY LLM-GENERATED ENVIRONMENTS MATTER:
- Demonstrates AI's ability to create training environments for AI
- Ensures no human bias in game design
- Optimized purely for RL training metrics
- Reproducible and deterministic by design

PROGRESSIVE DIFFICULTY SYSTEM:
The three-level structure enables curriculum learning, where agents
gradually develop from basic motor skills to complex predictive behaviors:

Level 1: Learn basic aiming without time pressure
Level 2: Add time management and prioritization  
Level 3: Master predictive targeting of moving objects

CONTROLS:
- W/A/S/D: Move crosshair (keys 5/7/8/9)
- E: Shoot at target (key 6)

SCORING:
- 1 point per successful hit
- Maximum 10 points per game
- Game ends after 10 targets appear

RL TRAINING NOTES:
- Reward: Delta of V[2] between frames
- Terminal state: When V[3] becomes 1
- Episode length: Variable based on player performance
- State space: 64x32 pixel display + register values
- Action space: 5 discrete actions

This environment demonstrates the potential for AI systems to create
their own training scenarios, potentially leading to self-improving
systems that can generate increasingly sophisticated challenges.

===================================================
""",
    "roms": {
        "target_shooter_level1": {
            "file": "target_shooter_level1.ch8",
            "description": "Static targets - Basic aiming skills",
            "platforms": ["originalChip8"]
        },
        "target_shooter_level2": {
            "file": "target_shooter_level2.ch8",
            "description": "Time-limited static targets - Adds urgency",
            "platforms": ["originalChip8"]
        },
        "target_shooter_level3": {
            "file": "target_shooter_level3.ch8",
            "description": "Moving time-limited targets - Maximum difficulty",
            "platforms": ["originalChip8"]
        }
    },
    "ai_generation_notes": {
        "model": "Large Language Model (LLM)",
        "year": "2024",
        "purpose": "RL Research Environment",
        "features": [
            "Deterministic mechanics",
            "Clear reward structure",
            "Progressive difficulty",
            "Observable state space",
            "Curriculum learning support"
        ]
    }
}