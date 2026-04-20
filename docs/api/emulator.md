# Emulator API

Low-level access to the CHIP-8 emulator. Most users will interact with [`OctaxEnv`](env.md) instead, but the emulator API is useful for debugging, custom wrappers, and extending the instruction set.

## State

```{eval-rst}
.. automodule:: octax.state
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```

### `EmulatorState` fields

| Field | Type | Description |
|---|---|---|
| `rng` | `PRNGKey` | JAX random key (consumed by `CXNN` random instruction) |
| `memory` | `uint8[4096]` | 4 KB address space |
| `pc` | `uint16` | Program counter — starts at `0x200` |
| `display` | `bool[64, 32]` | Monochrome pixel buffer (width × height) |
| `stack` | `StackState` | 16-entry subroutine return-address stack |
| `delay_timer` | `uint8` | Counts down at 60 Hz |
| `sound_timer` | `uint8` | Counts down at 60 Hz; triggers buzzer while > 0 |
| `keypad` | `bool[16]` | Key-pressed state for CHIP-8 keys 0–F |
| `V` | `uint8[16]` | General-purpose registers V0–VF |
| `I` | `uint16` | Index register |
| `modern_mode` | `bool` | Static flag — selects quirk behaviour for shift/load/jump |

## Core Functions

```{eval-rst}
.. automodule:: octax.emulator
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```

### `create_state`

```python
octax.create_state(rng: PRNGKey = jax.random.PRNGKey(0)) -> EmulatorState
```

Returns a fresh `EmulatorState` with the built-in font data pre-loaded at address `0x50`.

### `load_rom`

```python
octax.load_rom(state: EmulatorState, filename: str) -> EmulatorState
```

Reads a `.ch8` file from disk and copies its bytes into `state.memory` starting at `0x200`. Returns the updated state.

### `fetch`

```python
octax.fetch(state: EmulatorState) -> tuple[EmulatorState, uint16]
```

Reads the 16-bit instruction at `state.pc`, increments `pc` by 2, and returns `(new_state, instruction_word)`.

### `decode`

```python
octax.decode(instruction: int) -> DecodedInstruction
```

Splits a 16-bit instruction word into its component nibbles.

| Field | Bits | Description |
|---|---|---|
| `raw` | 15:0 | Original instruction word |
| `opcode` | 15:12 | First nibble — instruction family |
| `x` | 11:8 | Second nibble — VX register index |
| `y` | 7:4 | Third nibble — VY register index |
| `n` | 3:0 | Fourth nibble — 4-bit immediate |
| `nn` | 7:0 | Last byte — 8-bit immediate |
| `nnn` | 11:0 | Last 12 bits — 12-bit address |

### `execute`

```python
octax.execute(state: EmulatorState, instruction: int) -> EmulatorState
```

Dispatches `instruction` to the appropriate handler and returns the updated state. This is the core of the emulator's execution cycle.

## Running Multiple Instructions

`OctaxEnv` uses a JAX `lax.scan` loop internally. You can replicate this for custom pipelines:

```python
import jax
from octax import create_state, fetch, execute

@jax.jit
def run_n(state, n: int):
    def step(state, _):
        state, instr = fetch(state)
        state = execute(state, instr)
        return state, state

    final_state, all_states = jax.lax.scan(step, state, length=n)
    return final_state, all_states

state = create_state()
# ... load ROM ...
final, history = run_n(state, 1000)
```

## Memory Map

| Address range | Contents |
|---|---|
| `0x000 – 0x04F` | Reserved (interpreter area) |
| `0x050 – 0x09F` | Built-in hexadecimal font (5 bytes × 16 characters) |
| `0x0A0 – 0x1FF` | Unused (safe for data storage) |
| `0x200 – 0xFFF` | ROM / program area |
