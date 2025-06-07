from octax import EmulatorState

rom_path = "../../c8games/BRIX"

def score_fn(state: EmulatorState):
    return state.V[5]

def terminated_fn(state: EmulatorState):
    return state.V[14] == 0

action_set = [4, 6]
