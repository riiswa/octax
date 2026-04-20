# Game Reference

Each environment is documented with a gameplay GIF, summary table, description, action mapping, and reward details. For shared interface documentation see the [Environments overview](index.md).

---

## Puzzle

---

### Blinky

<div class="game-header">
<img src="../_static/imgs/blinky.gif" alt="Blinky gameplay"/>
</div>

This environment is part of the [Puzzle environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `blinky` |
| **Action Space** | `Discrete(5)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[6]` |
| **Termination** | `V[3] == 255` |
| **Release** | 1991 · Hans Christian Egeberg |

#### Description

A Pac-Man clone. Navigate a maze eating pills while avoiding two ghosts (Packlett and Heward). Ghost intelligence increases after each completed screen. The maze contains one left-right gateway and four energy pills near the corners. Two lives.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 3 | Move down |
| 1 | 6 | Move right |
| 2 | 7 | Move left |
| 3 | 8 | Move up |
| 4 | — | No-op |

#### Reward

Score delta from `V[6]` — points awarded for pills, energy pills, catching ghosts, and completing screens.

---

### Tetris

<div class="game-header">
<img src="../_static/imgs/tetris.gif" alt="Tetris gameplay"/>
</div>

This environment is part of the [Puzzle environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `tetris` |
| **Action Space** | `Discrete(5)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[10]` (lines cleared) |
| **Termination** | `V[1] == 2` (stack overflow) |
| **Release** | 1991 · Fran Dachille |

#### Description

Classic Tetris. Rotate and place falling tetrominoes to clear lines. Speed increases every 5 cleared lines, capping at 45. The game ends when the stack reaches the top of the playfield.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 4 | Rotate piece |
| 1 | 5 | Move left |
| 2 | 6 | Move right |
| 3 | 7 | Soft drop |
| 4 | — | No-op |

#### Reward

`+1` per line cleared. Score delta from `V[10]`.

---

### Worm

<div class="game-header">
<img src="../_static/imgs/worm.gif" alt="Worm gameplay"/>
</div>

This environment is part of the [Puzzle environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `worm` |
| **Action Space** | `Discrete(5)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[5]` |
| **Termination** | `V[7] == 255` |
| **Release** | 2007 · RB, Martijn Wenting / Revival Studios |

#### Description

Snake. Guide the worm to eat food pellets; each pellet grows the worm longer. The episode ends on collision with the worm's own body or the walls.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 2 | Move up |
| 1 | 8 | Move down |
| 2 | 4 | Move left |
| 3 | 6 | Move right |
| 4 | — | No-op |

#### Reward

`+1` per food pellet eaten. Score delta from `V[5]`.

---

## Action

---

### Brix

<div class="game-header">
<img src="../_static/imgs/brix.gif" alt="Brix gameplay"/>
</div>

This environment is part of the [Action environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `brix` |
| **Action Space** | `Discrete(3)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[5]` |
| **Termination** | `V[14] == 4` |
| **Release** | 1990 · Andreas Gustafsson |

#### Description

A Breakout clone. Bounce a ball upward to destroy bricks using a paddle at the bottom. 5 lives; a life is lost each time the ball drops below the paddle.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 4 | Move paddle left |
| 1 | 6 | Move paddle right |
| 2 | — | No-op |

#### Reward

`+1` per brick destroyed. Score delta from `V[5]`.

---

### Filter

<div class="game-header">
<img src="../_static/imgs/filter.gif" alt="Filter gameplay"/>
</div>

This environment is part of the [Action environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `filter` |
| **Action Space** | `Discrete(3)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[14]` |
| **Termination** | `V[13] == 0` |

#### Description

Catch drops falling from a pipe at the top of the screen with your paddle. Missing drops reduces your chances until the game ends.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 4 | Move paddle left |
| 1 | 6 | Move paddle right |
| 2 | — | No-op |

#### Reward

Score delta from `V[14]`.

---

### Pong

<div class="game-header">
<img src="../_static/imgs/pong.gif" alt="Pong gameplay"/>
</div>

This environment is part of the [Action environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `pong` |
| **Action Space** | `Discrete(3)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `(V[14]//10) − (V[14]%10)` |
| **Termination** | Either player reaches 9 points |
| **Note** | `disable_delay=True` |

#### Description

Single-player Pong against a CPU opponent. Score is the difference between the player's points (tens digit of `V[14]`) and the opponent's (units digit). The game ends when either side reaches 9.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 1 | Move paddle up |
| 1 | 4 | Move paddle down |
| 2 | — | No-op |

#### Reward

`+1` when player scores, `−1` when opponent scores.

---

### Squash

<div class="game-header">
<img src="../_static/imgs/squash.gif" alt="Squash gameplay"/>
</div>

This environment is part of the [Action environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `squash` |
| **Action Space** | `Discrete(3)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[11]` |
| **Termination** | `V[11] == 0` |
| **Release** | 1997 · David Winter |

#### Description

Bounce a ball around a squash court using a vertically-moving paddle on the right wall. The episode ends when the ball escapes past the paddle.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 1 | Move paddle up |
| 1 | 4 | Move paddle down |
| 2 | — | No-op |

#### Reward

Score delta from `V[11]`.

---

### Vertical Brix

<div class="game-header">
<img src="../_static/imgs/vertical_brix.gif" alt="Vertical Brix gameplay"/>
</div>

This environment is part of the [Action environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `vertical_brix` |
| **Action Space** | `Discrete(3)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[8]` |
| **Termination** | `V[7] == 0` |
| **Release** | 1996 · Paul Robson |
| **Note** | `disable_delay=True`; custom startup simulates title-skip key press |

#### Description

Breakout variant with vertically-placed bricks and a vertically-moving paddle on the left edge of the screen.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 1 | Move paddle up |
| 1 | 4 | Move paddle down |
| 2 | — | No-op |

#### Reward

Score delta from `V[8]`.

---

### Wipe Off

<div class="game-header">
<img src="../_static/imgs/wipe_off.gif" alt="Wipe Off gameplay"/>
</div>

This environment is part of the [Action environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `wipe_off` |
| **Action Space** | `Discrete(3)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[6]` |
| **Termination** | `V[7] == 0` |
| **Author** | Joseph Weisbecker |

#### Description

Clear all spots from the screen by bouncing a ball off a paddle. 20 balls total; from the original COSMAC VIP manual.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 4 | Move paddle left |
| 1 | 6 | Move paddle right |
| 2 | — | No-op |

#### Reward

`+1` per spot cleared. Score delta from `V[6]`.

---

## Strategy

---

### Missile Command

<div class="game-header">
<img src="../_static/imgs/missile.gif" alt="Missile Command gameplay"/>
</div>

This environment is part of the [Strategy environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `missile` |
| **Action Space** | `Discrete(2)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[7]` |
| **Termination** | `V[6] == 0` |
| **Release** | 1996 · David Winter |

#### Description

Hit 8 targets with 12 missiles. The shooter moves faster with each fired missile, making later shots harder to aim. Optimal play requires hitting all 8 targets within budget. Each hit scores 5 points.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 8 | Fire |
| 1 | — | No-op |

#### Reward

`+5` per target hit. Score delta from `V[7]`.

---

### Rocket

<div class="game-header">
<img src="../_static/imgs/rocket.gif" alt="Rocket gameplay"/>
</div>

This environment is part of the [Strategy environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `rocket` |
| **Action Space** | `Discrete(2)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[1]` |
| **Termination** | `V[2] == 9` |
| **Release** | 1978 · Joseph Weisbecker |

#### Description

A UFO moves across the top of the screen. Launch rockets from a random ground position to hit it. Nine rockets in total; each hit scores 1 point.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | F (15) | Launch rocket |
| 1 | — | No-op |

#### Reward

`+1` per UFO hit. Score delta from `V[1]`.

---

### Submarine

<div class="game-header">
<img src="../_static/imgs/submarine.gif" alt="Submarine gameplay"/>
</div>

This environment is part of the [Strategy environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `submarine` |
| **Action Space** | `Discrete(2)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[7]` |
| **Termination** | `V[8] == 0` |
| **Release** | 1978 · Carmelo Cortez |

#### Description

Drop depth charges on submarines moving below. Small sub: 15 points. Large sub: 5 points. 25 depth charges available.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 5 | Drop depth charge |
| 1 | — | No-op |

#### Reward

Score delta from `V[7]`.

---

### Tank Battle

<div class="game-header">
<img src="../_static/imgs/tank.gif" alt="Tank Battle gameplay"/>
</div>

This environment is part of the [Strategy environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `tank` |
| **Action Space** | `Discrete(6)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[14]` |
| **Termination** | `V[6] == 0` |
| **Release** | 197x |
| **Note** | `disable_delay=True`; directions 2 and 8 are swapped per original CHIP-8 keyboard |

#### Description

Navigate a tank and fire at a mobile target. 25 bombs; touching the target costs 5 bombs. Maximise hits before running out.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 2 | Move (swapped axis — see note) |
| 1 | 4 | Move left |
| 2 | 5 | Fire |
| 3 | 6 | Move right |
| 4 | 8 | Move (swapped axis — see note) |
| 5 | — | No-op |

#### Reward

Score delta from `V[14]`.

---

### UFO

<div class="game-header">
<img src="../_static/imgs/ufo.gif" alt="UFO gameplay"/>
</div>

This environment is part of the [Strategy environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `ufo` |
| **Action Space** | `Discrete(4)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[7]` |
| **Termination** | `V[8] == 0` |
| **Release** | 1992 · Lutz V |

#### Description

Fire in three directions from a stationary launcher at two UFOs flying at varying speeds. 15 missiles.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 4 | Fire left diagonal |
| 1 | 5 | Fire straight up |
| 2 | 6 | Fire right diagonal |
| 3 | — | No-op |

#### Reward

Score delta from `V[7]`.

---

## Exploration

---

### Cavern

<div class="game-header">
<img src="../_static/imgs/cavern1.gif" alt="Cavern gameplay"/>
</div>

This environment is part of the [Exploration environments](index.html#game-categories).

| | |
|---|---|
| **IDs** | `cavern1` … `cavern7` |
| **Action Space** | `Discrete(5)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[0]` (leftward progress) |
| **Termination** | `V[14] == 0` |
| **Release** | 2014 · Matthew Mikolay |
| **Note** | Modified ROM — leftward-progress reward replaces survival scoring |

#### Description

Escape a cave by navigating leftward through a maze without touching walls. Seven levels of increasing difficulty. The reward is non-decreasing: `+1` each time the agent reaches a new leftward-most X position, encouraging genuine exploration over oscillation.

```python
env, _ = create_environment("cavern3")  # Level 3
```

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 2 | Move up |
| 1 | 4 | Move left |
| 2 | 6 | Move right |
| 3 | 8 | Move down |
| 4 | — | No-op |

#### Reward

`+1` per new leftmost X reached. Score delta from `V[0]`.

---

### Flight Runner

<div class="game-header">
<img src="../_static/imgs/flight_runner.gif" alt="Flight Runner gameplay"/>
</div>

This environment is part of the [Exploration environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `flight_runner` |
| **Action Space** | `Discrete(5)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[7]` (walls cleared) |
| **Termination** | `V[5] == 255` or `V[7] == 255` |
| **Release** | 2014 · TodPunk |

#### Description

Pilot a ship through a procedurally scrolling corridor. Walls shift position as they scroll; score increments each time a wall segment cycles off the left edge.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 5 | Move up |
| 1 | 7 | Move left |
| 2 | 8 | Move down |
| 3 | 9 | Move right |
| 4 | — | No-op |

#### Reward

Score delta from `V[7]`.

---

### Space Flight

<div class="game-header">
<img src="../_static/imgs/space_flight1.gif" alt="Space Flight gameplay"/>
</div>

This environment is part of the [Exploration environments](index.html#game-categories).

| | |
|---|---|
| **IDs** | `space_flight1` … `space_flight10` |
| **Action Space** | `Discrete(3)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[0]` (frames survived) |
| **Termination** | `V[9] == 0` or `V[12] >= 0x3E` |
| **Note** | Modified ROM — single life, immediate game-over on collision |

#### Description

Navigate a spacecraft through asteroid fields. Ten levels with increasing asteroid density. Score is frames survived; the episode ends immediately on any collision.

```python
env, _ = create_environment("space_flight5")
```

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 1 | Move up |
| 1 | 4 | Move down |
| 2 | — | No-op |

#### Reward

`+1` per frame survived. Score delta from `V[0]`.

---

### Spacejam!

<div class="game-header">
<img src="../_static/imgs/spacejam.gif" alt="Spacejam! gameplay"/>
</div>

This environment is part of the [Exploration environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `spacejam` |
| **Action Space** | `Discrete(5)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[9]` |
| **Termination** | `V[10] == 0` |
| **Release** | 2015 · WilliamDonnelly |
| **Note** | `disable_delay=True` |

#### Description

Fly a ship through a scrolling corridor with dynamic walls and moving star obstacles. The corridor narrows over time; avoiding obstacles scores bonus points.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 5 | Move up |
| 1 | 8 | Move down |
| 2 | 7 | Move left |
| 3 | 9 | Move right |
| 4 | — | No-op |

#### Reward

Score delta from `V[9]` (time survival + obstacle avoidance bonuses).

---

## Shooter

---

### Airplane

<div class="game-header">
<img src="../_static/imgs/airplane.gif" alt="Airplane gameplay"/>
</div>

This environment is part of the [Shooter environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `airplane` |
| **Action Space** | `Discrete(2)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `−V[11] − V[12]` |
| **Termination** | `V[11] == 0` or `V[12] == 6` |
| **Release** | 19xx |

#### Description

A blitz-style bombing game. Fly over targets and drop bombs. Score is based on targets hit minus level-progression penalties.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 8 | Drop bomb |
| 1 | — | No-op |

#### Reward

Score delta from `−V[11] − V[12]`.

---

### Deep8

<div class="game-header">
<img src="../_static/imgs/deep.gif" alt="Deep8 gameplay"/>
</div>

This environment is part of the [Shooter environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `deep` |
| **Action Space** | `Discrete(4)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[9]` |
| **Termination** | `V[0xB] != 1` |
| **Release** | 2014 · John Earnest |
| **Note** | Custom startup key press to skip title |

#### Description

Defend your boat from incoming squid. Move left/right to dodge them and drop bombs — then detonate when a squid is in range. Too many squid tips the boat and ends the episode.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 7 | Move left |
| 1 | 8 | Drop bomb / detonate |
| 2 | 9 | Move right |
| 3 | — | No-op |

#### Reward

Score delta from `V[9]`.

---

### Shooting Stars

<div class="game-header">
<img src="../_static/imgs/shooting_stars.gif" alt="Shooting Stars gameplay"/>
</div>

This environment is part of the [Shooter environments](index.html#game-categories).

| | |
|---|---|
| **ID** | `shooting_stars` |
| **Action Space** | `Discrete(5)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[0]` (capped at 128 to handle uint8 overflow) |
| **Termination** | Never (truncated by `max_num_steps_per_episodes`) |
| **Release** | 1978 · Philip Baltzer |
| **Note** | `disable_delay=True` |

#### Description

Move a crosshair to intercept stars moving across the screen. No terminal state — runs until episode length is exceeded.

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 2 | Move up |
| 1 | 8 | Move down |
| 2 | 4 | Move left |
| 3 | 6 | Move right |
| 4 | — | No-op |

#### Reward

Score delta from `V[0]`.

---

## LLM-Generated Environments

---

### Target Shooter

<div class="game-header">
<img src="../_static/imgs/target_shooter1.gif" alt="Target Shooter gameplay"/>
</div>

This environment was designed entirely by an LLM for RL curriculum learning research.

| | |
|---|---|
| **IDs** | `target_shooter1` / `target_shooter2` / `target_shooter3` |
| **Action Space** | `Discrete(6)` |
| **Observation Space** | `Box(False, True, (4, 32, 64), bool)` |
| **Score Register** | `V[2]` |
| **Termination** | `V[3] == 1` (10 targets appeared) |
| **Release** | 2024 · LLM-generated |

#### Description

Move a crosshair and shoot at targets. Three levels implement a RL curriculum:

| Level | ID | Targets | Time limit |
|---|---|---|---|
| 1 | `target_shooter1` | Static | None |
| 2 | `target_shooter2` | Static | ~3 s |
| 3 | `target_shooter3` | Moving + bouncing | ~4 s |

The game ends after 10 total targets (hit or missed). Level 1 teaches basic aiming; Level 3 requires predictive interception of moving targets.

```python
env, _ = create_environment("target_shooter2")
```

#### Actions

| Action | Key | Meaning |
|---|---|---|
| 0 | 5 (W) | Move crosshair up |
| 1 | 7 (A) | Move crosshair left |
| 2 | 8 (S) | Move crosshair down |
| 3 | 9 (D) | Move crosshair right |
| 4 | 6 (E) | Shoot |
| 5 | — | No-op |

#### Reward

`+1` per target hit. Score delta from `V[2]`.
