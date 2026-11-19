"""Zero-continuation baseline that ranks candidates solely by one-step shaped rewards."""

import numpy as np


class PotentialAgent:
    def score_candidates(self, candidates):
        return np.asarray(candidates.learning_rewards, dtype=np.float32)

    def act(self, candidates):
        scores = self.score_candidates(candidates)
        return int(np.argmax(np.where(candidates.legal, scores, -np.inf)))
