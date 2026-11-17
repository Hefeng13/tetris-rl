import numpy as np


class RandomAgent:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def act(self, candidates):
        return int(self.rng.choice(np.flatnonzero(candidates.legal)))
