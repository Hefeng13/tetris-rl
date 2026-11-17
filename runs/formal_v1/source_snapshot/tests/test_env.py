import copy

import numpy as np
import pytest

from tetris_rl.env import PlacementEnv
from tetris_rl.features import board_statistics, candidate_features


@pytest.mark.parametrize("seed", [0, 7, 23])
def test_candidates_match_engine_at_multiple_board_heights(seed):
    env = PlacementEnv(piece_cap=100)
    candidates = env.reset(seed)
    rng = np.random.default_rng(seed)
    checked = 0
    for step in range(40):
        if step % 5 == 0:
            original = env.engine.board.copy()
            for action in np.flatnonzero(candidates.legal):
                reference = copy.deepcopy(env.engine)
                column, rotation = divmod(int(action), 4)
                for _ in range(rotation):
                    reference.active_tetromino = reference.rotate(reference.active_tetromino)
                reference.x = column + reference.padding - reference.active_tetromino.matrix.shape[1] // 2
                reference.y = 0
                _, reward, terminated, _, info = reference.step(reference.actions.hard_drop)
                assert reward == candidates.rewards[action]
                assert terminated == candidates.terminated[action]
                assert info["lines_cleared"] == candidates.lines[action]
                np.testing.assert_array_equal(reference.crop_padding(reference.board), candidates.boards[action])
                checked += 1
            np.testing.assert_array_equal(env.engine.board, original)
        action = int(rng.choice(np.flatnonzero(candidates.legal)))
        candidates, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    assert checked > 50


def test_line_clear_empty_board_is_valid_and_not_terminal():
    env = PlacementEnv()
    env.reset(0)
    env.engine.active_tetromino = env.engine.tetrominoes[1]
    padding = env.engine.padding
    env.engine.board[-padding - 2:-padding, padding:padding + 10] = 2
    env.engine.board[-padding - 2:-padding, padding:padding + 2] = 0
    candidates = env.candidates()
    action = 4
    assert candidates.legal[action]
    assert candidates.lines[action] == 2
    assert not candidates.terminated[action]
    assert np.all(candidates.features[action, :3] == 0)
    _, reward, terminated, _, info = env.step(action)
    assert reward == 41
    assert info["lines_cleared"] == 2
    assert not terminated


def test_all_terminal_candidates_can_end_the_game():
    env = PlacementEnv()
    env.reset(0)
    env.engine.board[:env.height, env.engine.padding:-env.engine.padding] = 2
    candidates = env.candidates()
    assert candidates.terminated[candidates.legal].all()
    _, reward, terminated, truncated, _ = env.step(int(np.flatnonzero(candidates.legal)[0]))
    assert terminated and not truncated and reward == -10


def test_truncation_retains_bootstrap_candidates_and_seed_replays():
    env = PlacementEnv(piece_cap=1)
    first = env.reset(11)
    action = int(np.flatnonzero(first.legal & ~first.terminated)[0])
    following, reward, terminated, truncated, _ = env.step(action)
    assert truncated and not terminated
    assert following is not None and following.legal.any()
    repeated = env.reset(11)
    np.testing.assert_array_equal(first.features, repeated.features)
    assert reward == 1


def test_feature_widths_and_signed_height_differences():
    boards = np.zeros((2, 20, 10), dtype=np.uint8)
    boards[0, -10:, 0] = 2
    boards[1, -5:, 1] = 2
    _, heights, holes, bumpiness = board_statistics(boards)
    np.testing.assert_array_equal(heights, [10, 5])
    np.testing.assert_array_equal(holes, [0, 0])
    np.testing.assert_array_equal(bumpiness, [10, 10])
    for dimension in (3, 13):
        result = candidate_features(boards, np.array([0, 1]), np.array([False, True]), dimension)
        assert result.shape == (2, dimension + 2)
        assert result.dtype == np.float32
