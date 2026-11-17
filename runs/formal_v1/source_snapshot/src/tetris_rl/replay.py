"""Replay stores next candidate sets, preserving action masks and truncation."""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, input_dim, action_count=40, seed=0):
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)
        self.features = np.empty((capacity, input_dim), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_features = np.zeros((capacity, action_count, input_dim), dtype=np.float32)
        self.next_legal = np.zeros((capacity, action_count), dtype=bool)
        self.terminated = np.empty(capacity, dtype=bool)

    def add(self, features, reward, next_candidates, terminated):
        index = self.position
        self.features[index] = features
        self.rewards[index] = reward
        self.terminated[index] = terminated
        self.next_features[index] = 0
        self.next_legal[index] = False
        if not terminated:
            if next_candidates is None or not next_candidates.legal.any():
                raise ValueError("A continuing transition needs legal next candidates")
            self.next_features[index] = next_candidates.features
            self.next_legal[index] = next_candidates.legal
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.rng.integers(self.size, size=batch_size)
        return tuple(torch.from_numpy(array[indices]) for array in (
            self.features, self.rewards, self.next_features, self.next_legal, self.terminated))
