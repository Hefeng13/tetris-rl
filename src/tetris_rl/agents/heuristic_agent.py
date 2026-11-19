import numpy as np

from ..features import board_statistics


class HeuristicAgent:
    def __init__(self, height_weight=-0.5, holes_weight=-0.36,
                 bumpiness_weight=-0.18, clear_weight=1.0):
        self.weights = height_weight, holes_weight, bumpiness_weight, clear_weight

    def score_candidates(self, candidates):
        _, maximum, holes, bumpiness = board_statistics(candidates.boards)
        height_weight, holes_weight, bumpiness_weight, clear_weight = self.weights
        score = (height_weight * maximum + holes_weight * holes
                 + bumpiness_weight * bumpiness + clear_weight * candidates.lines ** 2)
        return np.where(candidates.terminated, -1e6, score)

    def immediate_step_scores(self, candidates):
        return np.where(candidates.terminated, -1e6, self.weights[3] * candidates.lines ** 2)

    def act(self, candidates):
        score = self.score_candidates(candidates)
        return int(np.argmax(np.where(candidates.legal, score, -np.inf)))
