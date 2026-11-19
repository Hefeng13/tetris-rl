import copy

import numpy as np
import pytest

from tetris_rl.agents.dqn_agent import DQNAgent
from tetris_rl.agents.lookahead import LookaheadAgent
from tetris_rl.agents.potential_agent import PotentialAgent
from tetris_rl.config import load_config, make_env
from tetris_rl.evaluate import evaluate_policy


@pytest.mark.parametrize("shaping_enabled", [False, True])
def test_next_piece_candidates_match_actual_step(shaping_enabled):
    config = load_config()
    env = make_env(config, board_dim=13, piece_cap=50, shaping_enabled=shaping_enabled)
    candidates = env.reset(19)
    rng = np.random.default_rng(19)
    checked = 0
    for _ in range(15):
        surviving = env.unique_legal_actions()
        surviving = surviving[~candidates.terminated[surviving]]
        if len(surviving) == 0:
            break
        batch = env.next_piece_candidates(candidates.boards[surviving])
        next_piece_id = env.engine.tetrominoes[int(env.engine.queue.get_queue()[0])].id
        unique_next = env._unique_actions[next_piece_id]
        for idx, action in enumerate(surviving[:5]):
            branch = copy.deepcopy(env)
            actual_next, _, terminated, _, _ = branch.step(int(action))
            assert not terminated
            np.testing.assert_array_equal(batch.boards[idx], actual_next.boards[unique_next])
            np.testing.assert_array_equal(batch.lines[idx], actual_next.lines[unique_next])
            np.testing.assert_array_equal(batch.terminated[idx], actual_next.terminated[unique_next])
            np.testing.assert_allclose(batch.rewards[idx], actual_next.rewards[unique_next])
            np.testing.assert_allclose(batch.learning_rewards[idx], actual_next.learning_rewards[unique_next])
            np.testing.assert_allclose(batch.features[idx], actual_next.features[unique_next], atol=1e-6)
            checked += 1
        step_action = int(rng.choice(surviving))
        candidates, _, terminated, truncated, _ = env.step(step_action)
        if terminated or truncated:
            break
    assert checked >= 10
    env.close()


def test_potential_agent_matches_zero_network_dqn():
    config = load_config()
    env = make_env(config, board_dim=3, piece_cap=30, shaping_enabled=True)
    zero_dqn = DQNAgent(5, seed=0)
    for parameter in zero_dqn.network.parameters():
        parameter.data.zero_()
    potential_agent = PotentialAgent()
    candidates = env.reset(7)
    for _ in range(20):
        action = potential_agent.act(candidates)
        assert action == zero_dqn.act(candidates)
        candidates, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()


def test_lookahead_agent_matches_bruteforce_two_ply_dqn():
    config = load_config()
    env = make_env(config, board_dim=13, piece_cap=30)
    agent = DQNAgent(15, seed=3)
    lookahead_agent = LookaheadAgent(agent, env)
    candidates = env.reset(42)
    for _ in range(10):
        chosen = lookahead_agent.act(candidates)
        best_score, best_actions = -np.inf, []
        for action in np.flatnonzero(candidates.legal):
            if candidates.terminated[action]:
                score = float(candidates.learning_rewards[action])
            else:
                branch = copy.deepcopy(env)
                next_cand, _, terminated, _, _ = branch.step(int(action))
                assert not terminated
                q2 = agent.score_candidates(next_cand)
                best_q2 = float(np.max(np.where(next_cand.legal, q2, -np.inf)))
                score = float(candidates.learning_rewards[action]) + env.gamma * best_q2
                branch.close()
            if score > best_score + 1e-6:
                best_score = score
                best_actions = [int(action)]
            elif abs(score - best_score) <= 1e-6:
                best_actions.append(int(action))
        assert chosen in best_actions
        candidates, _, terminated, truncated, _ = env.step(chosen)
        if terminated or truncated:
            break
    env.close()


def test_classic_score_and_clear_counts_consistency():
    config = load_config()
    agent = PotentialAgent()
    rows, summary = evaluate_policy(agent, config, board_dim=3, seeds=range(40000, 40003),
                                    piece_cap=100, experiment="potential", lookahead=1)
    assert len(rows) == 3
    for row in rows:
        expected_lines = row["clears_1"] + 2 * row["clears_2"] + 3 * row["clears_3"] + 4 * row["clears_4"]
        expected_classic = 40 * row["clears_1"] + 100 * row["clears_2"] + 300 * row["clears_3"] + 1200 * row["clears_4"]
        assert row["lines"] == expected_lines
        assert row["classic_score"] == expected_classic
    assert np.isclose(summary["mean_classic_score"], np.mean([r["classic_score"] for r in rows]))


def test_custom_classic_learning_rewards_scale_and_preserve_engine_rewards():
    from pathlib import Path
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "final_classic_double.yaml")
    exp = config["experiments"][0]
    env = make_env(config, board_dim=13, piece_cap=50, shaping_enabled=False, experiment=exp)
    env.reset(0)
    env.engine.active_tetromino = env.engine.tetrominoes[1]
    padding = env.engine.padding
    env.engine.board[-padding - 2:-padding, padding:padding + 10] = 2
    env.engine.board[-padding - 2:-padding, padding:padding + 2] = 0
    candidates = env.candidates()
    action = 4
    assert candidates.lines[action] == 2
    assert candidates.rewards[action] == 41.0
    assert np.isclose(candidates.learning_rewards[action], (5.0 + 100.0) / 1000.0)
    _, reward, terminated, _, info = env.step(action)
    assert reward == candidates.rewards[action]
    assert info["lines_cleared"] == 2
    assert not terminated
    env.close()
