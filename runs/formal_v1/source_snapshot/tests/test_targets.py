import numpy as np
import torch

from tetris_rl.agents.dqn_agent import DQNAgent, td_targets
from tetris_rl.agents.heuristic_agent import HeuristicAgent
from tetris_rl.config import load_config, make_env
from tetris_rl.replay import ReplayBuffer
from tetris_rl.network import QNetwork


def test_dqn_and_double_dqn_respect_mask_and_termination():
    rewards = torch.tensor([1.0, -1.0, 2.0])
    terminated = torch.tensor([False, True, False])
    legal = torch.tensor([[True, False, True], [False, False, False], [True, True, False]])
    target = torch.tensor([[2.0, 100.0, 4.0], [5.0, 5.0, 5.0], [3.0, 1.0, 100.0]])
    online = torch.tensor([[9.0, 100.0, 1.0], [0.0, 0.0, 0.0], [1.0, 9.0, 100.0]])
    torch.testing.assert_close(td_targets(rewards, terminated, legal, target, 0.5),
                               torch.tensor([3.0, -1.0, 3.5]))
    torch.testing.assert_close(td_targets(rewards, terminated, legal, target, 0.5, online),
                               torch.tensor([2.0, -1.0, 2.5]))


def test_replay_wraparound_and_truncated_transition():
    env = make_env(load_config(), piece_cap=1)
    candidates = env.reset(3)
    action = HeuristicAgent().act(candidates)
    following, reward, terminated, truncated, _ = env.step(action)
    assert truncated and not terminated
    replay = ReplayBuffer(2, 5)
    for _ in range(3):
        replay.add(candidates.features[action], reward, following, terminated)
    assert replay.size == 2 and replay.position == 1
    assert replay.next_legal[:2].any(axis=1).all()
    assert not replay.terminated[:2].any()


def test_model_save_load_and_update(tmp_path):
    torch.set_num_threads(1)
    env = make_env(load_config())
    candidates = env.reset(0)
    agent = DQNAgent(5, seed=4)
    replay = ReplayBuffer(100, 5)
    for _ in range(32):
        action = agent.act(candidates, epsilon=1)
        following, reward, terminated, truncated, _ = env.step(action)
        replay.add(candidates.features[action], reward / 10, following, terminated)
        candidates = env.reset(0) if terminated or truncated else following
    assert np.isfinite(agent.update(replay, 16))
    path = tmp_path / "model.pt"
    agent.save(path, {"seed": 4})
    restored, metadata = DQNAgent.load(path)
    assert restored.act(candidates) == agent.act(candidates)
    assert metadata == {"seed": 4}


def test_greedy_validation_preserves_exploration_rng():
    env = make_env(load_config())
    candidates = env.reset(17)
    agent = DQNAgent(5, seed=8)
    state = agent.rng.bit_generator.state
    agent.act(candidates)
    assert agent.rng.bit_generator.state == state


def test_known_reward_is_counted_once_and_terminal_continuation_is_zero():
    network = QNetwork(5, gamma=0.95)
    for parameter in network.parameters():
        parameter.data.zero_()
    network.layers[-1].bias.data.fill_(2.0)
    features = torch.tensor([[0.0, 0.0, 0.0, 4.1, 0.0],
                             [0.0, 0.0, 0.0, -1.0, 1.0]])
    torch.testing.assert_close(network(features), torch.tensor([6.0, -1.0]))


def test_potential_shaping_uses_zero_potential_at_death():
    config = load_config()
    env = make_env(config, shaping_enabled=True)
    env.reset(2)
    env.engine.board[:env.height, env.engine.padding:-env.engine.padding] = 2
    candidates = env.candidates()
    # Full board has maximum height=20 and no holes, hence phi=-1.
    # Terminal shaping adds -phi(current)=+1 to the scaled game-over reward=-1.
    np.testing.assert_allclose(candidates.learning_rewards[candidates.legal], 0.0)
