"""One-step lookahead wrapper over the next known tetromino in the queue."""

import numpy as np


class LookaheadAgent:
    def __init__(self, base_agent, env, gamma=None):
        self.base_agent = base_agent
        self.env = env
        self.gamma = env.gamma if gamma is None else gamma

    def act(self, candidates):
        unique_legal = self.env.unique_legal_actions()
        surviving = unique_legal[~candidates.terminated[unique_legal]]
        if len(surviving) == 0:
            return self.base_agent.act(candidates)
        immediate_fn = getattr(self.base_agent, "immediate_step_scores", None)
        step1_immediate = (immediate_fn(candidates) if immediate_fn is not None
                           else candidates.learning_rewards)
        step2 = self.env.next_piece_candidates(candidates.boards[surviving])
        step2_scores = self.base_agent.score_candidates(step2)
        best_next = np.max(np.where(step2.legal, step2_scores, -np.inf), axis=1)
        action_scores = np.full(candidates.legal.shape, -np.inf, dtype=np.float64)
        terminal_actions = unique_legal[candidates.terminated[unique_legal]]
        if len(terminal_actions):
            action_scores[terminal_actions] = step1_immediate[terminal_actions]
        action_scores[surviving] = step1_immediate[surviving] + self.gamma * best_next
        return int(np.argmax(action_scores))
