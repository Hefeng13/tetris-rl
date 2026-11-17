"""Candidate Q-learning with consistent one-step reward indexing."""

import copy
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..network import QNetwork


def td_targets(rewards, terminated, next_legal, target_values, gamma, online_values=None):
    """Truncation continues to bootstrap. True termination suppresses it."""
    masked_target = target_values.masked_fill(~next_legal, -1e9)
    if online_values is None:
        future = masked_target.max(dim=1).values
    else:
        selected = online_values.masked_fill(~next_legal, -1e9).argmax(dim=1)
        future = masked_target.gather(1, selected[:, None]).squeeze(1)
    future = torch.where(terminated, torch.zeros_like(future), future)
    return rewards + gamma * future


class DQNAgent:
    def __init__(self, input_dim, seed=0, learning_rate=0.001, gamma=0.95, double=False,
                 known_reward=True, device="cpu"):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.network = QNetwork(input_dim, gamma, known_reward).to(self.device)
        self.target = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.double = double
        self.input_dim = input_dim

    def act(self, candidates, epsilon=0.0):
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(candidates.legal)))
        with torch.no_grad():
            features = torch.from_numpy(candidates.features).to(self.device)
            values = self.network(features).cpu().numpy()
        return int(np.argmax(np.where(candidates.legal, values, -np.inf)))

    def update(self, replay, batch_size):
        features, rewards, next_features, next_legal, terminated = (
            tensor.to(self.device) for tensor in replay.sample(batch_size))
        with torch.no_grad():
            target_values = self.target(next_features)
            online_values = self.network(next_features) if self.double else None
            targets = td_targets(rewards, terminated, next_legal, target_values,
                                 self.gamma, online_values)
        predicted = self.network(features)
        loss = F.smooth_l1_loss(predicted, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite TD loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.detach())

    def sync_target(self):
        self.target.load_state_dict(self.network.state_dict())

    def save(self, path, metadata):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"state_dict": self.network.state_dict(), "input_dim": self.input_dim,
                    "gamma": self.gamma, "known_reward": self.network.known_reward,
                    "metadata": metadata}, temporary)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=True)
        agent = cls(payload["input_dim"], gamma=payload.get("gamma", 0.95),
                    known_reward=payload.get("known_reward", False), device=device)
        agent.network.load_state_dict(payload["state_dict"])
        agent.sync_target()
        return agent, payload["metadata"]
