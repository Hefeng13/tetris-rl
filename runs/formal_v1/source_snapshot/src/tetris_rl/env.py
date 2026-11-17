"""Placement decisions using Tetris Gymnasium's game engine.

The candidate simulator computes known one-step consequences in parallel.
Actual transitions use the installed engine. Equivalence is regression-tested.
Terminal candidates remain available under the environment's legal-action mask.
"""

from dataclasses import dataclass

import numpy as np
from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.mappings.rewards import RewardsMapping

from .features import candidate_features, potential


@dataclass
class Candidates:
    features: np.ndarray
    legal: np.ndarray
    terminated: np.ndarray
    lines: np.ndarray
    rewards: np.ndarray
    boards: np.ndarray
    learning_rewards: np.ndarray


class PlacementEnv:
    def __init__(self, board_dim=3, piece_cap=500, width=10, height=20,
                 alife=1, clear_line=1, game_over=-10, reward_scale=10.0,
                 gamma=0.95, shaping=None):
        if width != 10:
            raise ValueError("The planned 3/13-feature comparison uses width=10")
        self.engine = Tetris(width=width, height=height,
                             rewards_mapping=RewardsMapping(
                                 alife=alife, clear_line=clear_line, game_over=game_over))
        self.board_dim = board_dim
        self.piece_cap = piece_cap
        self.width = width
        self.height = height
        self.action_count = width * 4
        self.reward_scale = reward_scale
        self.gamma = gamma
        self.shaping = shaping
        self.pieces = 0
        self.finished = False
        self._geometry = {}
        for piece in self.engine.tetrominoes:
            rows, columns = [], []
            for column in range(width):
                for rotation in range(4):
                    matrix = np.rot90(piece.matrix, rotation)
                    row, col = np.nonzero(matrix)
                    rows.append(row)
                    columns.append(col + column + self.engine.padding - matrix.shape[1] // 2)
            rows, columns = np.array(rows), np.array(columns)
            legal = ((columns >= self.engine.padding)
                     & (columns < self.engine.padding + width)).all(axis=1)
            self._geometry[piece.id] = (rows, columns, legal)

    def reset(self, seed):
        self.engine.reset(seed=int(seed))
        self.pieces = 0
        self.finished = False
        return self.candidates()

    @property
    def board(self):
        return self.engine.crop_padding(self.engine.board).copy()

    def candidates(self):
        if self.finished:
            raise RuntimeError("Reset the environment before requesting another decision")
        engine = self.engine
        rows, columns, legal = self._geometry[engine.active_tetromino.id]
        offsets = np.arange(self.height + 1)[None, :, None]
        collisions = (engine.board[rows[:, None, :] + offsets,
                                   columns[:, None, :]] != 0).any(axis=-1)
        first_collision = collisions.argmax(axis=1)
        landing_y = np.maximum(first_collision - 1, 0)
        immediate_death = collisions[:, 0]
        boards = np.repeat(engine.board[None], self.action_count, axis=0)
        placeable = np.flatnonzero(legal & ~immediate_death)
        boards[placeable[:, None], rows[placeable] + landing_y[placeable, None],
               columns[placeable]] = engine.active_tetromino.id
        interior = boards[:, :self.height, engine.padding:engine.padding + self.width]
        full_rows = (interior != 0).all(axis=-1)
        lines = full_rows.sum(axis=-1).astype(np.int64)
        lines[~legal | immediate_death] = 0
        for action in np.flatnonzero(lines):
            kept = interior[action, ~full_rows[action]].copy()
            interior[action] = 0
            interior[action, lines[action]:] = kept
        next_piece = engine.tetrominoes[engine.queue.get_queue()[0]]
        next_rows, next_cols = np.nonzero(next_piece.matrix)
        spawn_x = engine.width_padded // 2 - next_piece.matrix.shape[0] // 2
        blocked_spawn = (boards[:, next_rows, next_cols + spawn_x] != 0).any(axis=-1)
        terminated = immediate_death | blocked_spawn
        rewards = np.where(terminated, engine.rewards.game_over,
                           engine.rewards.alife + lines ** 2 * self.width).astype(np.float32)
        learning_rewards = rewards / self.reward_scale
        if self.shaping:
            before = potential(self.board, self.shaping["maximum_weight"], self.shaping["holes_weight"])
            after = potential(interior, self.shaping["maximum_weight"], self.shaping["holes_weight"])
            after = np.where(terminated, 0.0, after)
            learning_rewards = learning_rewards + self.gamma * after - before
        features = candidate_features(interior, learning_rewards, terminated, self.board_dim)
        return Candidates(features, legal.copy(), terminated, lines, rewards, interior.copy(),
                          learning_rewards.astype(np.float32))

    def step(self, action):
        if self.finished:
            raise RuntimeError("Reset the environment after termination or truncation")
        if not 0 <= int(action) < self.action_count:
            raise ValueError("Action outside the discrete placement space")
        column, rotation = divmod(int(action), 4)
        engine = self.engine
        _, _, legal = self._geometry[engine.active_tetromino.id]
        if not legal[action]:
            raise ValueError("The selected action collides with the frame")
        piece = engine.active_tetromino
        for _ in range(rotation):
            piece = engine.rotate(piece)
        engine.active_tetromino = piece
        engine.x = column + engine.padding - piece.matrix.shape[1] // 2
        engine.y = 0
        _, reward, terminated, _, info = engine.step(engine.actions.hard_drop)
        self.pieces += 1
        truncated = self.pieces >= self.piece_cap and not terminated
        # External truncation still exposes the next decision for bootstrapping.
        next_candidates = None if terminated else self.candidates()
        self.finished = bool(terminated or truncated)
        return next_candidates, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.engine.close()
