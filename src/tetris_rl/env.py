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
                 gamma=0.95, shaping=None, learning_line_scores=None,
                 learning_alife=None, learning_game_over=None):
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
        self.learning_line_scores = (np.asarray(learning_line_scores, dtype=np.float32)
                                     if learning_line_scores is not None else None)
        self.learning_alife = learning_alife
        self.learning_game_over = learning_game_over
        self.pieces = 0
        self.finished = False
        self._geometry = {}
        self._unique_actions = {}
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
            seen, unique = set(), []
            for action in np.flatnonzero(legal):
                signature = tuple(sorted(zip(rows[action].tolist(), columns[action].tolist())))
                if signature not in seen:
                    seen.add(signature)
                    unique.append(int(action))
            self._geometry[piece.id] = (rows, columns, legal)
            self._unique_actions[piece.id] = np.array(unique, dtype=np.int64)

    def reset(self, seed):
        self.engine.reset(seed=int(seed))
        self.pieces = 0
        self.finished = False
        return self.candidates()

    @property
    def board(self):
        return self.engine.crop_padding(self.engine.board).copy()

    def _pad_boards(self, interior_boards):
        padded = np.ones((interior_boards.shape[0], self.height + self.engine.padding,
                          self.width + 2 * self.engine.padding), dtype=interior_boards.dtype)
        pad = self.engine.padding
        padded[:, :self.height, pad:pad + self.width] = interior_boards
        return padded

    def _simulate(self, padded_boards, piece_id, next_queue_index, unique_only=False):
        engine = self.engine
        rows, columns, legal = self._geometry[piece_id]
        if unique_only:
            indices = self._unique_actions[piece_id]
            rows, columns, legal = rows[indices], columns[indices], legal[indices]
        action_count = len(legal)
        offsets = np.arange(self.height + 1)
        collisions = (padded_boards[:, rows[:, None, :] + offsets[None, :, None],
                                    columns[:, None, :]] != 0).any(axis=-1)
        first_collision = collisions.argmax(axis=-1)
        landing_y = np.maximum(first_collision - 1, 0)
        immediate_death = collisions[:, :, 0]
        boards = np.repeat(padded_boards[:, None], action_count, axis=1)
        placeable_b, placeable_a = np.nonzero(legal[None, :] & ~immediate_death)
        if len(placeable_b):
            boards[placeable_b[:, None], placeable_a[:, None],
                   rows[placeable_a] + landing_y[placeable_b, placeable_a][:, None],
                   columns[placeable_a]] = piece_id
        interior = boards[:, :, :self.height, engine.padding:engine.padding + self.width]
        full_rows = (interior != 0).all(axis=-1)
        lines = full_rows.sum(axis=-1).astype(np.int64)
        lines[:, ~legal] = 0
        lines[immediate_death] = 0
        clear_b, clear_a = np.nonzero(lines > 0)
        if len(clear_b):
            sub = interior[clear_b, clear_a]
            sub_full = full_rows[clear_b, clear_a]
            sub_lines = lines[clear_b, clear_a]
            order = np.argsort(~sub_full, axis=1, kind="stable")
            cleared = np.take_along_axis(sub, order[:, :, None], axis=1)
            cleared[np.arange(self.height)[None, :] < sub_lines[:, None]] = 0
            interior[clear_b, clear_a] = cleared
        next_piece = engine.tetrominoes[next_queue_index]
        next_rows, next_cols = np.nonzero(next_piece.matrix)
        spawn_x = engine.width_padded // 2 - next_piece.matrix.shape[0] // 2
        blocked_spawn = (boards[:, :, next_rows, next_cols + spawn_x] != 0).any(axis=-1)
        terminated = immediate_death | blocked_spawn
        rewards = np.where(terminated, engine.rewards.game_over,
                           engine.rewards.alife + lines ** 2 * self.width).astype(np.float32)
        if self.learning_line_scores is not None:
            alife_val = engine.rewards.alife if self.learning_alife is None else self.learning_alife
            over_val = engine.rewards.game_over if self.learning_game_over is None else self.learning_game_over
            raw_learning = np.where(terminated, over_val, alife_val + self.learning_line_scores[lines]).astype(np.float32)
            learning_rewards = raw_learning / self.reward_scale
        else:
            learning_rewards = rewards / self.reward_scale
        if self.shaping:
            before_boards = padded_boards[:, :self.height, engine.padding:engine.padding + self.width]
            before = potential(before_boards, self.shaping["maximum_weight"],
                               self.shaping["holes_weight"])[:, None]
            after = potential(interior, self.shaping["maximum_weight"], self.shaping["holes_weight"])
            after = np.where(terminated, 0.0, after)
            learning_rewards = learning_rewards + self.gamma * after - before
        features = candidate_features(interior, learning_rewards, terminated, self.board_dim)
        legal_batch = np.broadcast_to(legal[None, :], (padded_boards.shape[0], action_count)).copy()
        return Candidates(features, legal_batch, terminated, lines, rewards, interior.copy(),
                          learning_rewards.astype(np.float32))

    def unique_legal_actions(self):
        return self._unique_actions[self.engine.active_tetromino.id]

    def next_piece_candidates(self, interior_boards):
        queue = self.engine.queue.get_queue()
        next_piece_id = self.engine.tetrominoes[int(queue[0])].id
        return self._simulate(self._pad_boards(interior_boards), next_piece_id, int(queue[1]),
                              unique_only=True)

    def candidates(self):
        if self.finished:
            raise RuntimeError("Reset the environment before requesting another decision")
        queue = self.engine.queue.get_queue()
        batch = self._simulate(self.engine.board[None], self.engine.active_tetromino.id,
                               int(queue[0]), unique_only=False)
        return Candidates(batch.features[0], batch.legal[0], batch.terminated[0], batch.lines[0],
                          batch.rewards[0], batch.boards[0], batch.learning_rewards[0])

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
