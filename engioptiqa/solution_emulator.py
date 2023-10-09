class SolutionEmulator:
    def __init__(self, energy, frequency, is_feasible, values):
        self.energy = energy
        self.frequency = frequency
        self.is_feasible = is_feasible
        self.values = values

    def __repr__(self):
        return f"Solution (energy='{self.energy}', frequency={self.frequency})"
    
    def __getstate__(self):
        state = {
            'energy': self.energy,
            'frequency': self.frequency,
            'is_feasible': self.is_feasible,
            'values': self.values
        }
        return state

    def __setstate__(self, state):
        self.energy = state['energy']
        self.frequency = state['frequency']
        self.is_feasible = state['is_feasible']
        self.values = state['values']